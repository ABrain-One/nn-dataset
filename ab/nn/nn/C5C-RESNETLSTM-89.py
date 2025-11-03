import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utils
# -----------------------------
def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02) -> None:
    """
    Lightweight truncated normal: sample normal then clamp to ±2*std.
    Good enough for embeddings/positional encodings.
    """
    with torch.no_grad():
        tensor.normal_(mean=mean, std=std)
        tensor.clamp_(min=mean - 2 * std, max=mean + 2 * std)


def supported_hyperparameters():
    return {'lr', 'momentum'}


# -----------------------------
# Core Blocks
# -----------------------------
class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    """
    ViT-style block (kept to respect your original classes).
    Not used directly since we rely on nn.TransformerEncoder for stability,
    but this stays available if you want to swap later.
    """
    def __init__(self, hidden_dim: int, mlp_dim: int, num_heads: int, dropout: float, attention_dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff_net = MLPBlock(hidden_dim, mlp_dim, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])
        x = x + self.drop2(self.ff_net(self.norm2(x)))
        return x


# -----------------------------
# Vision Transformer Encoder
# -----------------------------
class VisionTransformerEncoder(nn.Module):
    """
    Simple/robust ViT encoder:
    - Conv2d patch-embedding
    - learned positional embeddings
    - nn.TransformerEncoder backbone
    Output: [B, N, H]
    """
    def __init__(
        self,
        in_shape: Tuple[int, int, int, int],
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        patch_size: int = 16,
    ):
        super().__init__()
        # Extract C,H,W from in_shape; allow both (B,C,H,W) or (C,H,W)
        if len(in_shape) == 4:
            _, c, h, w = in_shape
        elif len(in_shape) == 3:
            c, h, w = in_shape
        else:
            raise ValueError("in_shape must be (B,C,H,W) or (C,H,W)")

        assert h % patch_size == 0 and w % patch_size == 0, "H and W must be divisible by patch_size"
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.num_patches = (h // patch_size) * (w // patch_size)

        # Patchify
        self.patch_embed = nn.Conv2d(c, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=True)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.empty(1, self.num_patches, hidden_dim))
        trunc_normal_(self.pos_embedding, std=0.02)

        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.patch_embed(x)              # [B, Hdim, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)     # [B, N, Hdim]
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.ln(x)
        return x


# -----------------------------
# Decoder Block (kept for API)
# -----------------------------
class DecoderBlock(nn.Module):
    """
    Kept to preserve your original class name.
    Not used directly—TransformerDecoder below uses nn.TransformerDecoderLayers.
    """
    def __init__(self, hidden_dim: int, mlp_dim: int, num_heads: int, dropout: float, attention_dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.ff_net = MLPBlock(hidden_dim, mlp_dim, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x)
        x = self.self_attn(x, x, x, attn_mask=mask, need_weights=False)[0]
        x = self.drop(x)
        y = self.norm2(x)
        y = self.cross_attn(y, memory, memory, need_weights=False)[0]
        y = self.drop(y)
        z = self.ff_net(y)
        return z


# -----------------------------
# Transformer Decoder (text)
# -----------------------------
class TransformerDecoder(nn.Module):
    """
    Text decoder with causal self-attn + cross-attn (image features as memory).
    Keeps your method name/signature:
        forward(self, inputs, hidden_state=None, features=None) -> (logits, hidden_state)
    """
    def __init__(self, vocab_size: int, hidden_dim: int, num_heads: int, num_layers: int, dropout: float, attention_dropout: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_len = 100

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.empty(1, self.max_len, hidden_dim))
        trunc_normal_(self.pos_embedding, std=0.02)

        self.dropout = nn.Dropout(dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        # upper-triangular mask with -inf above diagonal
        mask = torch.full((T, T), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        inputs: torch.Tensor,                              # [B, T]
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        features: Optional[torch.Tensor] = None            # [B, N, H]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T = inputs.shape
        x = self.embedding(inputs) + self.pos_embedding[:, :T, :]
        x = self.dropout(x)

        tgt_mask = self._causal_mask(T, x.device)
        memory = features

        x = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)
        x = self.ln(x)
        logits = self.lm_head(x)                           # [B, T, V]

        # hidden_state placeholder to keep signature compatibility
        if hidden_state is None:
            h0 = torch.zeros(1, 1, self.hidden_dim, device=x.device)
            c0 = torch.zeros(1, 1, self.hidden_dim, device=x.device)
            hidden_state = (h0, c0)
        return logits, hidden_state


# -----------------------------
# Net (public API)
# -----------------------------
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        # ---- API aliases (auto-injected) ----
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3

        # try to infer vocab size robustly from out_shape variants
        if isinstance(out_shape, (tuple, list)):
            # common patterns: (V,), ((V,), something), [[V], ...], or nested
            def _first_int(x):
                if isinstance(x, int):
                    return x
                if isinstance(x, (tuple, list)) and len(x) > 0:
                    return _first_int(x[0])
                return None
            vs = _first_int(out_shape)
            if vs is None:
                raise ValueError(f"Cannot infer vocab_size from out_shape={out_shape}")
            self.vocab_size = int(vs)
        else:
            self.vocab_size = int(out_shape)

        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # Backward-compat locals (as in your snippet)
        vocab_size = self.vocab_size
        out_dim = self.vocab_size
        num_classes = self.vocab_size
        in_channels = self.in_channels

        # Hyperparams (allow override via prm)
        self.hidden_dim = int(prm.get('hidden_dim', 768))
        self.num_heads = int(prm.get('num_heads', 8))
        self.mlp_dim = int(prm.get('mlp_dim', 3072))
        self.dropout = float(prm.get('dropout', 0.1))
        self.attention_dropout = float(prm.get('attention_dropout', 0.1))
        self.enc_layers = int(prm.get('enc_layers', 6))
        self.dec_layers = int(prm.get('dec_layers', 6))
        self.patch_size = int(prm.get('patch_size', 16))
        self.max_len = int(prm.get('max_len', 50))

        # ---- Encoder ----
        enc_in_shape = in_shape if len(in_shape) == 4 else (1, self.in_channels, in_shape[-2], in_shape[-1])
        self.encoder = VisionTransformerEncoder(
            in_shape=enc_in_shape,
            hidden_dim=self.hidden_dim,
            num_layers=self.enc_layers,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            patch_size=self.patch_size,
        )

        # ---- Decoder ----
        self.decoder = TransformerDecoder(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.dec_layers,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
        )

        # training helpers
        self.criterion = None
        self.optimizer = None
        self.to(self.device)

    # ----------------- Training hooks -----------------
    def train_setup(self, prm: dict) -> None:
        """
        Set optimizer & loss. Expects prm to include 'lr' and optional 'momentum' (used as beta1).
        """
        lr = float(prm.get('lr', 1e-4))
        beta1 = float(prm.get('momentum', 0.9))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data) -> None:
        """
        One pass over a dataloader yielding (images, captions).
        Captions shape: [B, T] with 0 as PAD, and a BOS/EOS scheme.
        """
        assert self.criterion is not None and self.optimizer is not None, "Call train_setup(prm) first."
        self.train()
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        for images, captions in train_data:
            images = images.to(self.device, non_blocking=True).float()
            captions = captions.to(self.device, non_blocking=True).long()

            inputs = captions[:, :-1]            # teacher-forcing
            targets = captions[:, 1:]            # next-token

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                feats = self.encoder(images)                         # [B, N, H]
                logits, _ = self.decoder(inputs, None, feats)        # [B, T-1, V]
                loss = self.criterion(logits.reshape(-1, logits.size(-1)),
                                      targets.reshape(-1))

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=3.0)
            scaler.step(self.optimizer)
            scaler.update()

    # ----------------- Inference / Forward -----------------
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Training/Eval (teacher forcing): if captions is provided -> returns logits [B, T-1, vocab]
        Inference (greedy): if captions is None -> returns logits for generated sequence [B, T, vocab]
        """
        images = images.to(self.device).float()
        feats = self.encoder(images)  # [B, N, H]

        if captions is not None:
            captions = captions.to(self.device).long()
            inputs = captions[:, :-1]
            logits, _ = self.decoder(inputs, hidden_state, feats)
            return logits

        # Inference (greedy decode)
        B = images.size(0)
        sos_id = 1  # adjust if your tokenizer uses a different BOS
        cur = torch.full((B, 1), sos_id, dtype=torch.long, device=self.device)
        all_logits = []

        for _ in range(self.max_len):
            logits, _ = self.decoder(cur, hidden_state, feats)  # [B, t, V]
            step_logits = logits[:, -1:, :]                     # last step
            all_logits.append(step_logits)
            next_ids = step_logits.argmax(dim=-1)               # [B, 1]
            cur = torch.cat([cur, next_ids], dim=1)

        return torch.cat(all_logits, dim=1)  # [B, T, V]
