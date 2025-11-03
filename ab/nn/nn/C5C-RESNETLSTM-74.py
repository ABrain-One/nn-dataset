import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


# -----------------------------
# Attention/Conv utilities
# -----------------------------
class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels: int, reduction: float = 0.5, spatial_kernel: int = 7):
        super().__init__()
        # make a safe integer reduction >= 1
        red = max(1, int(round(channels * max(1e-6, reduction))))
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, red, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(red, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        ca = self.channel_mlp(x)
        x = x * ca

        # Spatial attention over [avg,max] maps
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial(torch.cat([avg_map, max_map], dim=1))
        return x * sa


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.depthwise.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.pointwise.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# -----------------------------
# Decoder components
# -----------------------------
class DecoderEmbeddings(nn.Module):
    """Token + sinusoidal positional embeddings."""
    def __init__(self, vocab_size: int, d_model: int, max_seq_length: int = 50, pad_idx: int = 0):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.register_buffer("pos_embed", self._build_sin_pos(max_seq_length, d_model), persistent=False)

    @staticmethod
    def _build_sin_pos(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [L, D]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T]
        B, T = tokens.size()
        tok = self.token_embed(tokens)                 # [B, T, D]
        pos = self.pos_embed[:T, :].unsqueeze(0)       # [1, T, D]
        return tok + pos


class DecoderBlock(nn.Module):
    """Self-attn -> Cross-attn -> FFN with pre-norm + residuals."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, causal_mask: Optional[torch.Tensor] = None):
        # x: [B, T, D], memory: [B, S, D]
        x_norm = self.ln1(x)
        sa, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=causal_mask, need_weights=False)
        x = x + self.dropout(sa)

        x_norm = self.ln2(x)
        ca, _ = self.cross_attn(x_norm, memory, memory, need_weights=False)
        x = x + self.dropout(ca)

        x_norm = self.ln3(x)
        x = x + self.dropout(self.ffn(x_norm))
        return x


class TransformerDecoder(nn.Module):
    """Stack of DecoderBlocks + LM head."""
    def __init__(self, d_model: int, num_heads: int, num_layers: int, vocab_size: int, pad_idx: int = 0, max_len: int = 50, dropout: float = 0.1):
        super().__init__()
        self.embed = DecoderEmbeddings(vocab_size, d_model, max_len, pad_idx)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.d_model = d_model

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # lower-triangular mask: True = allowed, False = masked (MultiheadAttention expects float mask or bool)
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        # convert to float mask where masked positions = -inf
        float_mask = mask.float().masked_fill(mask, float("-inf"))
        return float_mask  # [T, T]

    def forward(self, memory_tokens: torch.Tensor, captions: Optional[torch.Tensor] = None, sos_idx: int = 1, eos_idx: int = 2) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        memory_tokens: [B, S, D]  (image tokens)
        captions: [B, T] or None  (teacher forcing when provided)
        Returns:
            logits [B, T, V] and (optionally) generated tokens when captions is None.
        """
        device = memory_tokens.device
        B = memory_tokens.size(0)

        if captions is not None:
            # teacher forcing
            T = captions.size(1)
            x = self.embed(captions)                            # [B, T, D]
            causal = self._causal_mask(T, device)               # [T, T] float
            for layer in self.layers:
                x = layer(x, memory_tokens, causal_mask=causal)
            x = self.ln(x)
            logits = self.fc_out(x)                             # [B, T, V]
            return logits, None

        # Greedy generation
        tokens = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)  # start with <SOS>
        logits_all = []
        for step in range(self.max_len):
            x = self.embed(tokens)                              # [B, t, D]
            causal = self._causal_mask(x.size(1), device)
            h = memory_tokens
            for layer in self.layers:
                x = layer(x, h, causal_mask=causal)
            x = self.ln(x)
            step_logits = self.fc_out(x[:, -1:, :])             # [B,1,V]
            logits_all.append(step_logits)
            next_tok = step_logits.argmax(-1)                   # [B,1]
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok.squeeze(1) == eos_idx).all():
                break
        logits = torch.cat(logits_all, dim=1) if logits_all else torch.zeros(B, 0, self.fc_out.out_features, device=device)
        return logits, tokens


# -----------------------------
# Net (Encoder + Decoder)
# -----------------------------
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: Dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = int(in_shape[1]) if isinstance(in_shape, (tuple, list)) and len(in_shape) > 1 else 3
        self.vocab_size = int(out_shape[0] if isinstance(out_shape, (tuple, list)) else out_shape)

        # Hyperparams
        self.hidden_size = int(prm.get("hidden_size", 768))
        self.num_layers = int(prm.get("num_layers", 4))
        self.num_heads  = int(prm.get("num_heads", 8))
        self.dropout    = float(prm.get("dropout", 0.1))
        self.max_len    = int(prm.get("max_len", 30))
        self.pad_idx    = int(prm.get("pad_idx", 0))
        self.sos_idx    = int(prm.get("sos_idx", 1))
        self.eos_idx    = int(prm.get("eos_idx", 2))

        if self.hidden_size % self.num_heads != 0:
            # make it divisible
            for h in (12, 8, 6, 4, 2):
                if self.hidden_size % h == 0:
                    self.num_heads = h
                    break

        # ----- Encoder (CNN -> tokens [B,S,D]) -----
        enc_channels = [64, 128, 256, 256]
        layers = []
        in_c = self.in_channels
        for i, c in enumerate(enc_channels):
            stride = 2 if i > 0 else 1
            layers += [
                DepthwiseSeparableConv2d(in_c, c, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                CBAM(c, reduction=0.25),
            ]
            in_c = c
        self.cnn = nn.Sequential(*layers)
        self.enc_proj = nn.Conv2d(enc_channels[-1], self.hidden_size, kernel_size=1, bias=False)

        # ----- Decoder -----
        self.decoder = TransformerDecoder(
            d_model=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            vocab_size=self.vocab_size,
            pad_idx=self.pad_idx,
            max_len=self.max_len,
            dropout=self.dropout,
        )

        # training helpers
        self.criteria: Tuple[nn.Module, ...] = ()
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    # --- training API ---
    def train_setup(self, prm: Dict):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm["lr"],
            betas=(prm.get("momentum", 0.9), 0.999),
        )

    def learn(self, train_data):
        """train_data: iterable of (images, captions) where captions is [B,T] or [B,1,T]"""
        if not self.criteria or self.optimizer is None:
            return
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            if captions.dim() == 3:  # [B,1,T] -> [B,T]
                captions = captions[:, 0, :]

            # teacher forcing: inputs=tokens[:-1], targets=tokens[1:]
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            # encoder
            mem = self._encode(images)               # [B,S,D]
            # decoder
            logits, _ = self.decoder(mem, inputs, self.sos_idx, self.eos_idx)  # [B,T-1,V]

            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # --- core forward ---
    def forward(self, images, captions=None, hidden_state=None):
        """
        If captions provided: returns (logits [B,T-1,V], None) using teacher forcing.
        If captions is None: returns (logits [B,Tgen,V], tokens) using greedy generation.
        """
        mem = self._encode(images.to(self.device))  # [B,S,D]
        if captions is not None:
            if captions.dim() == 3:
                captions = captions[:, 0, :]
            captions = captions.to(self.device)
            if captions.size(1) <= 1:
                empty = torch.zeros(captions.size(0), 0, self.vocab_size, device=self.device)
                return empty, None
            inputs = captions[:, :-1]
            logits, _ = self.decoder(mem, inputs, self.sos_idx, self.eos_idx)
            return logits, None
        else:
            logits, tokens = self.decoder(mem, None, self.sos_idx, self.eos_idx)
            return logits, tokens

    # --- helpers ---
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B,C,H,W] -> cnn -> [B,C',H',W'] -> project -> [B,D,H',W'] -> tokens [B,S,D]
        x = self.cnn(images)
        x = self.enc_proj(x)                         # [B,D,H',W']
        B, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, D)  # [B,S,D]
        return x


# ---- quick self-test (optional) ----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W, V = 2, 3, 224, 224, 5000
    in_shape = (B, C, H, W)
    out_shape = (V,)
    prm = dict(lr=1e-4, momentum=0.9, hidden_size=768, num_layers=2, num_heads=8, dropout=0.1,
               max_len=16, pad_idx=0, sos_idx=1, eos_idx=2)

    net = Net(in_shape, out_shape, prm, device)
    net.train_setup(prm)

    imgs = torch.randn(B, C, H, W, device=device)
    caps = torch.randint(0, V, (B, 12), device=device)
    caps[:, 0] = prm["sos_idx"]

    # Teacher forcing
    tf_logits, _ = net(imgs, caps)
    print("Teacher-forcing logits:", tuple(tf_logits.shape))  # (B, T-1, V)

    # Greedy decode
    gen_logits, tokens = net(imgs, captions=None)
    print("Greedy logits:", tuple(gen_logits.shape), "Tokens:", tokens.shape if tokens is not None else None)
