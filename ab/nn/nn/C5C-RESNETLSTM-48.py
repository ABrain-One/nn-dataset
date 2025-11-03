import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


class SqueezeAndExcite(nn.Module):
    """Squeeze-and-Excitation (safe types & dims)."""

    def __init__(self, input_channels: int, se_ratio: float = 0.5) -> None:
        super().__init__()
        # If se_ratio < 1 treat as fractional shrink; else treat as integer divisor.
        if se_ratio < 1.0:
            reduced = max(1, int(round(input_channels * se_ratio)))
        else:
            reduced = max(1, input_channels // int(round(se_ratio)))

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, input_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.squeeze(x)
        w = self.fc(w).view(x.size(0), x.size(1), 1, 1)
        return x * w


class VisionTransformerEncoder(nn.Module):
    """
    Minimal encoder that returns (B, S, E) features.
    Signature matches usage in Net; uses lazy convs so we don't need in_channels at init.
    """

    def __init__(
        self,
        vision_layers: int = 12,
        embed_dim: int = 768,
        num_heads: int = 8,
        qkv_proj_groups: Optional[int] = None,
        dpr_dropout_prob: float = 0.1,
        global_pool: str = "avg",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.trunk = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(self.embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1) if global_pool == "avg" else nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)             # [B, E, H', W']
        h = self.pool(h)              # [B, E, 1, 1]
        h = torch.flatten(h, 1)       # [B, E]
        h = h.unsqueeze(1)            # [B, 1, E] -> single "image token"
        return h


class Net(nn.Module):
    """Main image captioning model combining an encoder-decoder pipeline."""

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # Extract vocab size robustly from possibly nested tuples/lists
        def _first_int(x):
            return _first_int(x[0]) if isinstance(x, (tuple, list)) else int(x)

        self.vocab_size = _first_int(out_shape)
        self.d_model = 768

        # Encoder
        self.vision_transformer_encoder = VisionTransformerEncoder(
            vision_layers=12,
            embed_dim=self.d_model,
            num_heads=8,
            qkv_proj_groups=None,
            dpr_dropout_prob=0.1,
            global_pool="avg",
        )

        # Decoder
        self.decoder_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model,
            padding_idx=0,
        )
        layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=3072,
            dropout=min(float(prm.get("dropout", 0.2)), 0.3),
            activation="gelu",
            batch_first=True,  # use (B, T, E)
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.output_proj = nn.Linear(self.d_model, self.vocab_size)

        # Tokens
        self.sos_idx = 1
        self.eos_idx = self.vocab_size - 1

        # Training helpers (can be overridden in train_setup)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(prm.get("lr", 1e-4)), betas=(float(prm.get("momentum", 0.9)), 0.999)
        )

        self.to(self.device)

    # API helper: provide a transformer-style "hidden state" placeholder
    def init_zero_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, batch_size, self.d_model, device=self.device),
            torch.zeros(1, batch_size, self.d_model, device=self.device),
        )

    @staticmethod
    def _subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        # (T, T) causal mask with -inf above diagonal for additive masking
        mask = torch.full((sz, sz), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Encode images -> memory: (B, S=1, E)
        memory = self.vision_transformer_encoder(images)

        if captions is not None:
            # Teacher forcing: predict next token for each position
            # inputs: all but last; targets: all but first
            inputs = captions[:, :-1]  # (B, T-1)
            tgt_emb = self.decoder_embedding(inputs)  # (B, T-1, E)

            tgt_mask = self._subsequent_mask(tgt_emb.size(1), self.device)  # (T-1, T-1)
            dec_out = self.decoder(tgt=tgt_emb, memory=memo ry, tgt_mask=tgt_mask)  # (B, T-1, E)
            logits = self.output_proj(dec_out)  # (B, T-1, V)
            return logits, hidden_state
        else:
            # Greedy decode
            B = images.size(0)
            ys = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)
            logits_steps = []
            for _ in range(20):  # max length
                tgt_emb = self.decoder_embedding(ys)  # (B, t, E)
                tgt_mask = self._subsequent_mask(tgt_emb.size(1), self.device)
                dec_out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)  # (B, t, E)
                step_logits = self.output_proj(dec_out[:, -1:, :])  # (B, 1, V)
                logits_steps.append(step_logits)
                next_ids = step_logits.argmax(dim=-1)  # (B, 1)
                ys = torch.cat([ys, next_ids], dim=1)
                if (next_ids == self.eos_idx).all():
                    break
            logits = torch.cat(logits_steps, dim=1) if logits_steps else torch.empty(B, 0, self.vocab_size, device=self.device)
            return logits, hidden_state

    # Optional training helpers to match expected API
    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get("lr", 1e-4)),
            betas=(float(prm.get("momentum", 0.9)), 0.999),
        )

    def learn(self, train_data) -> None:
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)

            logits, _ = self.forward(images, captions)  # (B, T-1, V)
            targets = captions[:, 1:]  # (B, T-1)

            loss = self.criteria[0](logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
