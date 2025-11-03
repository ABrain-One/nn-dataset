import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Harness hook (module-level)
def supported_hyperparameters():
    return {"lr", "momentum"}


# ---- Utility blocks (not required by forward path, but kept and fixed) --------
class EfficientChannelReduction(nn.Module):
    def __init__(self, input_channels: int, num_reduced_channels: Optional[int]):
        super().__init__()
        if num_reduced_channels is None or num_reduced_channels <= 0:
            num_reduced_channels = max(1, input_channels // 4)
        self.act = nn.ReLU(inplace=True)
        self.reduce = nn.Linear(input_channels, num_reduced_channels)

    def forward(self, x: Tensor) -> Tensor:
        # Expect x: [*, C]; apply linear reduction safely
        x = self.act(x)
        return self.reduce(x)


class ChannelShuffle(nn.Module):
    """Channel Shuffle; unused in forward but kept compile-safe."""
    def __init__(self, groups: int):
        super().__init__()
        self.groups = max(1, int(groups))

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.size()
        if self.groups <= 1 or c % self.groups != 0:
            return x
        g = self.groups
        x = x.view(b, g, c // g, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, c, h, w)
        return x


# ---- Attention block ----------------------------------------------------------
class DecoderAttention(nn.Module):
    """Cross-attention: queries attend over memory (keys=values=memory)."""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(
        self,
        queries: Tensor,              # [B, Tq, E]
        memory: Tensor,               # [B, Tm, E]
        key_padding_mask: Optional[Tensor] = None,  # [B, Tm] True => ignore
    ) -> Tuple[Tensor, Tensor]:
        out, w = self.multihead_attn(queries, memory, memory, key_padding_mask=key_padding_mask)
        return out, w


# ---- Main model ---------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.prm = prm or {}

        # Hidden / vocab
        self.hidden_size = int(self.prm.get("hidden_size", 768))
        self.num_heads = int(self.prm.get("num_heads", 8))
        assert self.hidden_size % self.num_heads == 0, "Hidden size must be divisible by number of heads"
        self.vocab_size = self._first_int(out_shape)
        self.pad_idx = int(self.prm.get("pad_idx", 0))
        self.sos_idx = int(self.prm.get("sos_idx", 1))

        # Infer channels / spatial dims safely
        self.in_channels, self.in_h, self.in_w = self._infer_shape(in_shape)

        # --- Image encoder -> token memory [B, S, H] ---
        # Small CNN to create hidden_size channels then flatten to tokens
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.Conv2d(128, self.hidden_size, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size), nn.ReLU(inplace=True),
        )

        # (optional) memory projections; kept to match original concept
        self.memory_proj_key = nn.Linear(self.hidden_size, self.hidden_size)
        self.memory_proj_value = nn.Linear(self.hidden_size, self.hidden_size)

        # --- Decoder: GRU + cross-attention + classifier ---
        embed_dim = int(self.prm.get("embed_dim", self.hidden_size))
        self.embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx=self.pad_idx)
        self.gru = nn.GRU(embed_dim, self.hidden_size, batch_first=True, num_layers=1)
        self.attention = DecoderAttention(self.hidden_size, self.num_heads)
        self.classifier = nn.Linear(self.hidden_size, self.vocab_size)

        # Regularization
        self.dropout = nn.Dropout(float(self.prm.get("dropout", 0.1)))

        # Init
        self._init_weights()
        self.to(self.device)

    # Shared by some harnesses
    @staticmethod
    def supported_hyperparameters():
        return {"lr", "momentum"}

    # ---- Training glue ---------------------------------------------------------
    def train_setup(self, prm):
        prm = prm or {}
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device),)
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        self.train_setup(getattr(train_data, "prm", self.prm))
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device).long()
            if captions.dim() == 3:
                captions = captions[:, 0, :]  # [B,1,T] -> [B,T]
            if captions.size(1) <= 1:
                continue

            # Teacher forcing: predict next token for positions 1..T-1
            inputs = captions[:, :-1]                  # [B, T-1]
            targets = captions[:, 1:]                  # [B, T-1]
            logits, _ = self.forward(images, inputs)   # [B, T-1, V]

            loss = self.criteria[0](logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # ---- Forward ---------------------------------------------------------------
    def forward(self, images: Tensor, captions: Optional[Tensor] = None):
        """
        If captions given (teacher-forced):
            images: [B, C, H, W], captions: [B, T]  -> logits: [B, T, V]
        If captions is None:
            one-step from SOS -> logits: [B, 1, V]
        Returns (logits, attention_weights)
        """
        images = images.to(self.device)

        # Encode image -> memory tokens [B, S, H]
        feat = self.encoder(images)                   # [B, Hdim, H', W']
        B, C, Hp, Wp = feat.shape
        memory = feat.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)   # [B, S, H]
        # Optional linear projections (keys/values share dim H)
        mem_k = self.memory_proj_key(memory)          # [B, S, H]
        mem_v = self.memory_proj_value(memory)        # [B, S, H]
        # MultiheadAttention uses a single memory for K and V; concatenate not needed
        # We'll pass mem_v as memory; projections are symmetrical so either works
        memory_tokens = mem_v

        # Build decoder inputs
        if captions is None:
            captions = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)
        else:
            captions = captions.long().to(self.device)

        emb = self.dropout(self.embedding(captions))  # [B, T, E]
        dec_out, _ = self.gru(emb)                    # [B, T, H]

        # Cross-attention: queries = decoder states, memory = image tokens
        attn_out, attn_w = self.attention(dec_out, memory_tokens, key_padding_mask=None)  # [B, T, H], [B, T, S]
        logits = self.classifier(self.dropout(attn_out))  # [B, T, V]
        return logits, attn_w

    # ---- Helpers ---------------------------------------------------------------
    def _init_weights(self):
        init_range = 0.02
        # Embedding
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_range)
        # Linear layers
        for m in [self.memory_proj_key, self.memory_proj_value, self.classifier]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # Conv/BN already have decent defaults

        # GRU init (only l0 present)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    @staticmethod
    def _first_int(x) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return Net._first_int(x[0])
        return int(x)

    @staticmethod
    def _infer_shape(in_shape: tuple) -> Tuple[int, int, int]:
        # Accept (C,H,W) or (N,C,H,W)
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) >= 4:  # (N, C, H, W)
                return int(in_shape[1]), int(in_shape[2]), int(in_shape[3])
            if len(in_shape) == 3:  # (C, H, W)
                return int(in_shape[0]), int(in_shape[1]), int(in_shape[2])
            if len(in_shape) == 2:  # (C, H) -> assume square
                return int(in_shape[0]), int(in_shape[1]), int(in_shape[1])
        return 3, 224, 224
