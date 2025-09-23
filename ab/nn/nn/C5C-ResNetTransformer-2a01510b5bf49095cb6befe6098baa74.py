import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.avg(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.proj is not None:
            identity = self.proj(identity)
        out = self.relu(out + identity)
        return out


class PositionalEncoding(nn.Module):
    """Sinusoidal PE for sequence (batch_first: [B, T, D])."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class PositionalEncoding2D(nn.Module):
    """Lightweight 2D PE for encoder memory tokens (assumes H=W)."""
    def __init__(self, d_model: int):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D PE"
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D] where S = H*W and H==W
        B, S, D = x.shape
        H = int(S ** 0.5)
        W = H
        assert H * W == S, "S must be a perfect square for 2D PE"

        device = x.device
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij'
        )
        grid = torch.stack([xx, yy], dim=-1).reshape(1, S, 2).float()  # [1,S,2]

        halfD = D // 2
        div = torch.exp(torch.arange(0, halfD, 2, device=device).float() * (-math.log(10000.0) / halfD))
        pe = torch.zeros(1, S, D, device=device)

        # x-pos
        pe[..., 0:halfD:2] = torch.sin(grid[..., 0:1] * div)
        pe[..., 1:halfD:2] = torch.cos(grid[..., 0:1] * div)
        # y-pos
        pe[..., halfD::2] = torch.sin(grid[..., 1:2] * div)
        pe[..., halfD+1::2] = torch.cos(grid[..., 1:2] * div)

        return x + pe


class Encoder(nn.Module):
    """
    Outputs multi-token memory [B, S, D] (S = tokens_hw^2).
    Using tokens_hw=7 -> S=49 tokens per image.
    """
    def __init__(self, in_channels: int, d_model: int = 768, tokens_hw: int = 7):
        super().__init__()
        c1, c2, c3 = 64, 128, 256
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 7, 2, 3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = BasicBlock(c1, c1, stride=1, use_se=True)
        self.layer2 = BasicBlock(c1, c2, stride=2, use_se=True)
        self.layer3 = BasicBlock(c2, c3, stride=2, use_se=True)

        self.tokens_hw = tokens_hw
        self.proj = nn.Conv2d(c3, d_model, kernel_size=1, bias=False)
        self.mem_pe = PositionalEncoding2D(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)                                           # [B, C3, H, W]
        x = F.adaptive_avg_pool2d(x, (self.tokens_hw, self.tokens_hw))
        x = self.proj(x)                                             # [B, D, h, w]
        B, D, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)                             # [B, S=h*w, D]
        x = self.mem_pe(x)
        return x                                                     # [B, S, D]


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 768, nhead: int = 8,
                 num_layers: int = 1, dim_ff: int = 2048, dropout: float = 0.2):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, vocab_size, bias=True)

        # Weight tying improves sample efficiency
        self.out.weight = self.embedding.weight

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def forward(self, tgt_tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        tgt_tokens: [B, T]
        memory:     [B, S, D]
        returns logits: [B, T, V]
        """
        x = self.embedding(tgt_tokens) * math.sqrt(self.d_model)  # [B,T,D]
        x = self.pos(x)
        mask = self._causal_mask(x.size(1), x.device)
        dec = self.decoder(tgt=x, memory=memory, tgt_mask=mask)   # [B,T,D]
        return self.out(dec)                                      # [B,T,V]

    @torch.no_grad()
    def beam_search(self, memory, start_id=1, end_id=2, max_len=50, beam_size=5, length_penalty=0.6):
        B = memory.size(0)
        beams = [(torch.full((B,1), start_id, device=memory.device, dtype=torch.long),
                torch.zeros(B, device=memory.device), 1)]  # (tokens, sum_logp, length)
        for _ in range(max_len):
            new_beams = []
            for tokens, logp, L in beams:
                x = self.embedding(tokens) * math.sqrt(self.d_model)
                x = self.pos(x)
                mask = self._causal_mask(x.size(1), x.device)
                dec = self.decoder(tgt=x, memory=memory, tgt_mask=mask)
                logits = self.out(dec[:, -1, :])                      # [B,V]
                next_logp = F.log_softmax(logits, dim=-1)             # [B,V]
                topk_logp, topk = next_logp.topk(beam_size, dim=-1)
                for k in range(beam_size):
                    new_tokens = torch.cat([tokens, topk[:, k:k+1]], dim=1)
                    new_logp   = logp + topk_logp[:, k]
                    new_beams.append((new_tokens, new_logp, L+1))
            # rank by length-penalized average logprob: lp / (len^α)
            def score(t):
                toks, lp, L = t
                denom = (L ** length_penalty)
                return (lp / denom).mean().item()
            new_beams.sort(key=score, reverse=True)
            beams = new_beams[:beam_size]
            if all((b[0][:, -1] == end_id).all().item() for b in beams):
                break
        best = max(beams, key=lambda t: (t[1] / (t[2] ** length_penalty)).mean().item())[0]
        return best[:, 1:]  # strip SOS



class Net(nn.Module):
    """
    CNN encoder → multi-token memory → Transformer decoder.
    Training path: returns logits Tensor [B, T-1, V].
    Inference path (captions=None): returns token ids [B, <=max_len].
    """
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        in_channels = int(in_shape[1])
        vocab_size = int(out_shape[0])

        d_model = int(prm.get('hidden_dim', 768))   # ≥640 recommended
        nhead = 8
        num_layers = int(prm.get('num_layers', 2))
        dropout = float(prm.get('dropout', 0.2))

        self.encoder = Encoder(in_channels, d_model=d_model, tokens_hw=8)
        self.decoder = TransformerDecoder(vocab_size, d_model=d_model, nhead=nhead,
                                          num_layers=num_layers, dropout=dropout)
        self.vocab_size = vocab_size
        self.sos_id = int(prm.get('sos_id', 1))
        self.eos_id = int(prm.get('eos_id', 2))

        # training components (set in train_setup)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.optimizer: Optional[torch.optim.Optimizer] = None

    # --- helpers ---
    @staticmethod
    def _norm_caps(caps: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if caps is None:
            return None
        if caps.ndim == 3:
            caps = caps[:, 0, :]  # [B,1,T] → [B,T]
        elif caps.ndim == 1:
            caps = caps.unsqueeze(0)
        return caps.long()

    # --- public API ---
    def train_setup(self, prm: dict):
        self.to(self.device)
        # LR floor + clamped beta1 sourced from "momentum"
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.8), 0.99)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = self.criterion.to(self.device)

    def learn(self, train_data):
        self.train()
        for images, caps in train_data:
            images = images.to(self.device, non_blocking=True)
            caps = self._norm_caps(caps.to(self.device, non_blocking=True))  # [B,T]

            # teacher forcing
            inputs = caps[:, :-1]       # [B,T-1]
            targets = caps[:, 1:]       # [B,T-1]

            memory = self.encoder(images)                 # [B,S,D]
            logits = self.decoder(inputs, memory)         # [B,T-1,V]

            # Sentry checks
            assert images.dim() == 4
            assert logits.shape[1] == inputs.shape[1]
            assert logits.shape[-1] == self.vocab_size

            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        memory = self.encoder(images.to(self.device, non_blocking=True))  # [B,S,D]
        if captions is not None:
            caps = self._norm_caps(captions).to(self.device)
            inputs = caps[:, :-1]
            logits = self.decoder(inputs, memory)                         # [B,T-1,V]
            return logits                                                 # Tensor only (BLEU expects .dim())
        # inference path: return tokens for evaluator that can handle ids
        return self.decoder.beam_search(memory, start_id=self.sos_id, end_id=self.eos_id,
                                        max_len=50, beam_size=5, length_penalty=0.6)  # [B,<=max_len]
