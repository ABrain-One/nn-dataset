import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


# --------------------- Positional encodings ---------------------
def sinusoidal_pos_enc(n: int, d: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(n, d, device=device)
    position = torch.arange(0, n, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# --------------------- Encoder (ViT-ish, patchify + Transformer) ---------------------
class ViTSpatialEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 768, patch_size: int = 16, depth: int = 6, nheads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # Patch embedding: [B, C, H, W] -> [B, D, H/ps, W/ps]
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # Learnable cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, _, H, W = x.shape
        x = self.patch_embed(x)                  # [B, D, Hp, Wp]
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)         # [B, N, D], N = Hp*Wp

        # add positional encodings (sinusoidal -> no resizing headaches)
        pos = sinusoidal_pos_enc(x.size(1), D, x.device)  # [N, D]
        x = x + pos.unsqueeze(0)                # [B, N, D]

        # prepend cls token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)          # [B, N+1, D]

        x = self.encoder(x)                     # [B, N+1, D]
        return x                                # return all tokens (cls + patches)


# --------------------- Decoder (Transformer) ---------------------
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 768, depth: int = 2, nheads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=nheads, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.dec_layer, num_layers=depth)
        self.proj = nn.Linear(hidden_size, vocab_size)

    @staticmethod
    def causal_mask(L: int, device: torch.device) -> torch.Tensor:
        # float mask with -inf in upper triangle (no peeking)
        mask = torch.full((L, L), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.masked_fill(mask != 0, float("-inf")).masked_fill(mask == 0, 0.0)
        return mask  # [L, L]

    def forward(self, captions: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # captions: [B, T], memory: [B, S, D]
        B, T = captions.shape
        x = self.embed(captions)  # [B, T, D]
        x = x + sinusoidal_pos_enc(T, self.hidden_size, x.device).unsqueeze(0)

        tgt_mask = self.causal_mask(T, x.device)   # [T, T]
        out = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)  # [B, T, D]
        logits = self.proj(out)  # [B, T, V]
        return logits


# --------------------- Full Net ---------------------
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm or {}

        def _vsize(shape) -> int:
            if isinstance(shape, int):
                return int(shape)
            if isinstance(shape, (tuple, list)) and len(shape) > 0:
                first = shape[0]
                if isinstance(first, (tuple, list)) and len(first) > 0:
                    return int(first[0])
                return int(first)
            return int(shape)

        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) and len(in_shape) > 1 else 3
        self.vocab_size = _vsize(out_shape)
        self.hidden_size = int(self.prm.get("hidden_size", 768))
        self.patch_size = int(self.prm.get("patch_size", 16))
        self.dec_depth = int(self.prm.get("dec_depth", 2))
        self.dec_heads = int(self.prm.get("dec_heads", 8))
        self.max_len = int(self.prm.get("max_len", 32))
        self.bos_id = int(self.prm.get("bos_id", 1))
        self.eos_id = int(self.prm.get("eos_id", 2))

        self.encoder = ViTSpatialEncoder(
            in_channels=self.in_channels, hidden_dim=self.hidden_size, patch_size=self.patch_size, depth=6, nheads=8
        )
        self.decoder = TransformerDecoder(
            vocab_size=self.vocab_size, hidden_size=self.hidden_size, depth=self.dec_depth, nheads=self.dec_heads
        )

        self.criteria = None
        self.optimizer = None
        self.to(self.device)

    # ---- Training helpers ----
    def train_setup(self, prm):
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        if self.criteria is None or self.optimizer is None:
            self.train_setup(self.prm)
        criterion = self.criteria[0]
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            # teacher forcing: predict next token
            inp = captions[:, :-1]
            tgt = captions[:, 1:]

            memory = self.encoder(images)                # [B, S, D]
            logits = self.decoder(inp, memory)           # [B, T, V]
            loss = criterion(logits.reshape(-1, self.vocab_size), tgt.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # ---- Forward (train + greedy inference) ----
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # kept for API compatibility
    ):
        images = images.to(self.device)
        memory = self.encoder(images)  # [B, S, D]

        if captions is not None:
            captions = captions.to(self.device)
            logits = self.decoder(captions, memory)
            return logits, None

        # Greedy decode
        B = images.size(0)
        cur = torch.full((B, 1), self.bos_id, dtype=torch.long, device=self.device)  # [B, 1]
        outputs = [cur]
        for _ in range(self.max_len - 1):
            logits = self.decoder(cur, memory)           # [B, t, V]
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)  # [B, 1]
            outputs.append(next_token)
            cur = torch.cat([cur, next_token], dim=1)
            if (next_token == self.eos_id).all():
                break
        seq = torch.cat(outputs, dim=1)                  # [B, T]
        return seq, None
