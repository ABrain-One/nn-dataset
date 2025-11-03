import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Iterable


# ----------------------------
# Public API: supported hparams
# ----------------------------
def supported_hyperparameters():
    return {'lr', 'momentum'}


# ----------------------------
# Helpers to parse shapes safely
# ----------------------------
def _infer_in_channels(in_shape: Union[Tuple[int, ...], Iterable[int]]) -> int:
    if not isinstance(in_shape, (tuple, list)) or len(in_shape) == 0:
        return 3
    if len(in_shape) >= 3:
        return int(in_shape[-3])   # (C,H,W) or (B,C,H,W) -> C
    return int(in_shape[0])


def _infer_vocab_size(out_shape) -> int:
    if isinstance(out_shape, int):
        return out_shape
    if isinstance(out_shape, (tuple, list)):
        for x in out_shape:
            try:
                v = _infer_vocab_size(x)
                if isinstance(v, int) and v > 0:
                    return v
            except Exception:
                continue
    raise ValueError(f"Could not infer vocab size from out_shape={out_shape!r}")


# ----------------------------
# Core building blocks
# ----------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, channel: int, ratio: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, max(1, channel // ratio))
        self.fc2 = nn.Linear(max(1, channel // ratio), channel)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        s = self.pool(x).view(b, c)        # [B, C]
        e = self.fc2(self.relu(self.fc1(s)))
        scale = self.sig(e).view(b, c, 1, 1)
        return x * scale


class PositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding (added, not learned)."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)        # [L, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [L, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)          # even
        pe[:, 1::2] = torch.cos(position * div_term)          # odd
        pe = pe.unsqueeze(0)                                   # [1, L, D]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return self.dropout(x + self.pe[:, :T])


class BagNetUnit(nn.Module):
    """Conv -> BN -> ReLU -> SE."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        x = self.se(x)
        return x


class CustomCrossAttention(nn.Module):
    """
    Cross-attention between decoder states (queries) and encoder memory.
    decoder_states: [B, T, H]
    encoder_states: [B, L, C]  (L = spatial locations)
    Returns: context [B, T, H], attn_weights [B, T, L]
    """
    def __init__(self, hidden_dim: int, mem_dim: int):
        super().__init__()
        d_k = hidden_dim
        self.w_q = nn.Linear(hidden_dim, d_k, bias=False)
        self.w_k = nn.Linear(mem_dim, d_k, bias=False)
        self.w_v = nn.Linear(mem_dim, hidden_dim, bias=False)
        self.scale = math.sqrt(d_k)

    def forward(self, decoder_states: torch.Tensor, encoder_states: torch.Tensor):
        # Q: [B,T,d_k], K: [B,L,d_k], V: [B,L,H]
        Q = self.w_q(decoder_states)
        K = self.w_k(encoder_states)
        V = self.w_v(encoder_states)

        # scores: [B,T,L]
        scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale
        weights = F.softmax(scores, dim=-1)

        # context: [B,T,H]
        context = torch.matmul(weights, V)
        return context, weights


class Decoder(nn.Module):
    """GRU decoder with embeddings."""
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embed_layer = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=(dropout if num_layers > 1 else 0.0))

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        # x: [B, T]
        embedded = self.embed_layer(x)        # [B, T, E]
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden


# ----------------------------
# Main network
# ----------------------------
class Net(nn.Module):
    """Encoder-Decoder with cross-attention for image captioning."""
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.prm = dict(
            embed_dim=int(prm.get('embed_dim', 512)),
            hidden_dim=int(prm.get('hidden_dim', 768)),
            num_layers=int(prm.get('num_layers', 1)),
            dropout=float(prm.get('dropout', 0.1)),
            sos_index=int(prm.get('sos_index', 1)),
            eos_index=int(prm.get('eos_index', 2)),
            max_len=int(prm.get('max_len', 30)),
        )
        self.device = device

        # Shapes
        in_channels = _infer_in_channels(in_shape)
        vocab_size = _infer_vocab_size(out_shape)
        self.vocab_size = vocab_size

        # -------- Encoder: small conv pyramid -> memory tokens [B, L, Cenc]
        enc_channels = [64, 128, 256]
        self.encoder = nn.Sequential(
            BagNetUnit(in_channels, enc_channels[0], 3, 1),
            BagNetUnit(enc_channels[0], enc_channels[0], 3, 1),
            BagNetUnit(enc_channels[0], enc_channels[1], 3, 2),  # /2
            BagNetUnit(enc_channels[1], enc_channels[1], 3, 1),
            BagNetUnit(enc_channels[1], enc_channels[2], 3, 2),  # /4
            BagNetUnit(enc_channels[2], enc_channels[2], 3, 1),
        )
        self.enc_out_dim = enc_channels[-1]  # mem_dim

        # -------- Decoder + cross-attention
        self.decoder = Decoder(vocab_size=vocab_size,
                               embed_dim=self.prm['embed_dim'],
                               hidden_dim=self.prm['hidden_dim'],
                               num_layers=self.prm['num_layers'],
                               dropout=self.prm['dropout'])
        self.posenc = PositionalEncoding(self.prm['hidden_dim'], max_len=512, dropout=self.prm['dropout'])
        self.cross_attention = CustomCrossAttention(hidden_dim=self.prm['hidden_dim'],
                                                    mem_dim=self.enc_out_dim)
        self.output_layer = nn.Linear(self.prm['hidden_dim'], vocab_size)

    # ---- Training boilerplate
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999)
        )

    def learn(self, train_data):
        """Expects iterable of (images, captions) with captions padded (pad=0) and having SOS/EOS."""
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)

            # Teacher forcing: in -> captions[:, :-1], targets -> captions[:, 1:]
            logits, _ = self.forward(images, captions)
            targets = captions[:, 1:].contiguous()
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

            # yield loss for external logging
            yield loss.detach()

    # ---- Forward (train + inference)
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Train: pass y (captions) -> returns (logits [B,T-1,V], hidden)
        Inference: y=None -> greedy decode -> returns (logits [B,T,V], hidden)
        """
        x = x.to(self.device)
        B = x.size(0)

        # ----- Encoder to memory: [B, L, Cenc]
        feats = self.encoder(x)                      # [B, Cenc, H', W']
        B, Cenc, Hp, Wp = feats.shape
        memory = feats.permute(0, 2, 3, 1).reshape(B, Hp * Wp, Cenc)  # [B, L, Cenc]

        # ----- Training (teacher forcing)
        if y is not None:
            if y.dim() == 3:
                y = y[:, 0, :]
            y_in = y[:, :-1].contiguous()           # [B, T-1]
            dec_out, hidden = self.decoder(y_in)    # [B, T-1, H], hidden
            dec_out = self.posenc(dec_out)

            context, _ = self.cross_attention(dec_out, memory)  # [B, T-1, H]
            fused = dec_out + context                           # residual
            logits = self.output_layer(fused)                   # [B, T-1, V]
            return logits, hidden

        # ----- Inference (greedy)
        sos = self.prm['sos_index']
        eos = self.prm['eos_index']
        max_len = self.prm['max_len']

        inputs = torch.full((B, 1), sos, dtype=torch.long, device=self.device)  # [B,1]
        hidden: Optional[torch.Tensor] = None
        logits_steps = []
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            dec_step, hidden = self.decoder(inputs[:, -1:], hidden)  # [B,1,H]
            dec_step = self.posenc(dec_step)
            context, _ = self.cross_attention(dec_step, memory)      # [B,1,H]
            fused = dec_step + context
            step_logits = self.output_layer(fused)                   # [B,1,V]
            logits_steps.append(step_logits)

            next_token = step_logits.argmax(dim=-1)                  # [B,1]
            inputs = torch.cat([inputs, next_token], dim=1)
            finished |= (next_token.squeeze(1) == eos)
            if finished.all():
                break

        logits = torch.cat(logits_steps, dim=1) if len(logits_steps) else torch.zeros(B, 0, self.vocab_size, device=self.device)
        return logits, hidden
