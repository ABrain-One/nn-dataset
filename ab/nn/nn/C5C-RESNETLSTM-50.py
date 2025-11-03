import math
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


# -------------------------
# Building blocks
# -------------------------
class CNBlock(nn.Module):
    """
    Lightweight MLP-style block with LayerNorm and residual.
    Works on inputs shaped (B, T, D) or (B, D) (it will treat the last dim as features).
    """
    def __init__(
        self,
        dim: int,
        layer_scale: float = 1e-6,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm1 = (norm_layer(dim) if norm_layer is not None else nn.LayerNorm(dim, eps=1e-6))
        self.linear1 = nn.Linear(dim, 4 * dim)
        self.linear2 = nn.Linear(4 * dim, dim)
        self.activation = nn.GELU()
        # layer scale as a learnable scalar (broadcast on last dim)
        self.ls = nn.Parameter(torch.tensor(float(layer_scale)), requires_grad=True)
        self.sd = float(stochastic_depth_prob)  # kept for compatibility; not applied here

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm1(x)
        y = self.activation(self.linear1(y))
        y = self.linear2(self.activation(y))
        return x + self.ls * y


class PositionalEncoding(nn.Module):
    """Batch-first sinusoidal positional encoding: expects input of shape (B, T, D)."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(1, max_len, d_model)  # (1, T, D)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=True)

    def forward(self, input_tensor: Tensor) -> Tensor:
        # input_tensor: (B, T, D)
        t = input_tensor.size(1)
        return self.dropout(input_tensor + self.pe[:, :t, :])


class MultiheadAttentionWrapper(nn.Module):
    """Simple wrapper around nn.MultiheadAttention using batch_first=True."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.multihead = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        return self.multihead(query=query, key=key, value=value, key_padding_mask=key_padding_mask, attn_mask=attn_mask)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation for 2D feature maps."""
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        red = max(1, dim // max(1, reduction))
        self.fc = nn.Sequential(
            nn.Linear(dim, red, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(red, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# -------------------------
# Main model
# -------------------------
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # Config
        self.in_channels = int(in_shape[1]) if len(in_shape) > 1 else 3
        # Robustly extract vocab size from possibly nested shapes
        def _first_int(x):
            return _first_int(x[0]) if isinstance(x, (tuple, list)) else int(x)
        self.vocab_size = _first_int(out_shape)

        self.hidden_size = int(prm.get("hidden_size", 640))
        self.num_attn_heads = int(prm.get("num_attn_heads", 8))
        self.encoder_architecture = str(prm.get("encoder_architecture", "resnet")).lower()
        self.add_se_module = bool(prm.get("add_se_module", False))
        self.seq_length = int(prm.get("seq_length", 20))
        self.grad_clipping_max = float(prm.get("grad_clip", 3.0))
        self.total_stages = 10  # used if ViT-like path is selected

        # ------- Encoder -------
        if self.encoder_architecture == "vit":
            # A very lightweight "ViT-ish" frontend (still conv, then token MLPs)
            self.encoder = nn.Sequential(
                nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                # Token MLP blocks working on a flattened token dimension; we end with a global head
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, self.hidden_size),
                CNBlock(self.hidden_size, layer_scale=1e-3),
            )
        else:
            # ResNet-ish encoder with optional SE
            layers: list[nn.Module] = [
                nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
            if self.add_se_module:
                layers.append(SEBlock(64))
            layers += [
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, self.hidden_size),
            ]
            self.encoder = nn.Sequential(*layers)

        # ------- Decoder -------
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.pos_encoding = PositionalEncoding(self.hidden_size, dropout=0.1, max_len=max(self.seq_length, 100))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_attn_heads,
            dim_feedforward=max(256, 2 * self.hidden_size),
            dropout=0.1,
            batch_first=True,  # (B, T, D)
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

        # ------- Optim stuff (can be reset in train_setup) -------
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get("lr", 1e-4)),
            betas=(float(prm.get("momentum", 0.9)), 0.999),
        )

        self.to(self.device)

    # ---- helpers ----
    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device) -> Tensor:
        # (T, T) with -inf above diagonal to enforce causality
        mask = torch.full((sz, sz), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def init_hidden_state(self, batch_size: int):
        # Transformer decoder doesn't use explicit hidden state
        return None, None

    # ---- training API ----
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(prm["lr"]), betas=(float(prm.get("momentum", 0.9)), 0.999)
        )

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            if isinstance(captions, tuple):
                captions = captions[0]

            images = images.to(self.device)
            captions = captions.to(self.device)

            # Normalize caption shape to (B, T)
            if captions.dim() == 3:
                captions = captions.argmax(dim=-1)

            logits, _ = self.forward(images, captions)

            targets = captions[:, 1:]  # next-token targets
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clipping_max)
            self.optimizer.step()

            # yield training loss (float)
            yield float(loss.detach().cpu().item())

    # ---- forward / decode ----
    def forward(
        self,
        images: Tensor,
        captions: Optional[Tensor] = None,
        hidden_state: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        """
        images: (B, C, H, W)
        captions: (B, T) with integer token ids (padding id=0). If provided, teacher forcing is used.
        Returns (logits, memory) where logits is (B, T-1, V) during teacher forcing.
        """
        # Encode images -> memory of shape (B, 1, D)
        feats = self.encoder(images)          # (B, D)
        memory = feats.unsqueeze(1)           # (B, 1, D)

        if captions is not None:
            # Teacher forcing: predict next token for each position
            inp = captions[:, :-1]            # (B, T-1)
            tgt = self.embedding(inp)         # (B, T-1, D)
            tgt = self.pos_encoding(tgt)      # (B, T-1, D)
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), tgt.device)  # (T-1, T-1)
            dec = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)              # (B, T-1, D)
            logits = self.fc_out(dec)          # (B, T-1, V)
            return logits, memory

        # Inference: greedy decode up to seq_length
        B = images.size(0)
        sos_idx = 1  # conventional SOS id
        eos_idx = self.vocab_size - 1  # assume last id is EOS if used
        ys = torch.full((B, 1), sos_idx, device=images.device, dtype=torch.long)

        logits_steps = []
        for _ in range(self.seq_length):
            tgt = self.embedding(ys)               # (B, T, D)
            tgt = self.pos_encoding(tgt)
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), tgt.device)
            dec = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)  # (B, T, D)
            step_logits = self.fc_out(dec[:, -1:, :])                      # (B, 1, V)
            logits_steps.append(step_logits)
            next_ids = step_logits.argmax(dim=-1)                           # (B, 1)
            ys = torch.cat([ys, next_ids], dim=1)
            if (next_ids == eos_idx).all():
                break

        logits = torch.cat(logits_steps, dim=1) if logits_steps else torch.empty(B, 0, self.vocab_size, device=images.device)
        return logits, memory
