import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02) -> None:
    """
    Simple truncated normal initializer: samples from N(mean, std) and clamps to 2*std.
    (Approximation suitable for embeddings/positional encodings.)
    """
    with torch.no_grad():
        tensor.normal_(mean=mean, std=std)
        tensor.clamp_(min=mean - 2 * std, max=mean + 2 * std)


def supported_hyperparameters():
    return {'lr', 'momentum'}


# -----------------------------
# Blocks
# -----------------------------
class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    """
    A ViT-style encoder block: LN -> MHA -> Drop -> Residual -> LN -> MLP -> Drop -> Residual
    (Kept for completeness; the encoder below uses nn.TransformerEncoder for stability.)
    """
    def __init__(self, hidden_dim: int, mlp_dim: int, num_heads: int, dropout: float, attention_dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = MLPBlock(hidden_dim, mlp_dim, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x = x + self.drop1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])
        # Feed-forward
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


# -----------------------------
# Vision Transformer Encoder
# -----------------------------
class VisionTransformerEncoder(nn.Module):
    """
    Lightweight ViT encoder implemented with a Conv2d patch-embedding + TransformerEncoder.
    Produces features of shape [B, N, hidden_dim], where N = (H/patch)*(W/patch).
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
        # in_shape is expected like (B, C, H, W) or similar; we only need C, H, W
        _, c, h, w = in_shape if len(in_shape) == 4 else (None, in_shape[0], in_shape[1], in_shape[2])
        assert h % patch_size == 0 and w % patch_size == 0, "H and W must be divisible by patch_size"

        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.num_patches = (h // patch_size) * (w // patch_size)

        # Patchify via strided convolution to hidden_dim channels
        self.patch_embed = nn.Conv2d(c, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=True)

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.empty(1, self.num_patches, hidden_dim))
        trunc_normal_(self.pos_embedding, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # A stable, battle-tested encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.patch_embed(x)                    # [B, hidden_dim, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)           # [B, N, hidden_dim]
        x = x + self.pos_embedding                 # add learned positions
        x = self.dropout(x)
        x = self.encoder(x)                        # [B, N, hidden_dim]
        x = self.norm(x)
        return x


# -----------------------------
# Transformer Decoder (text)
# -----------------------------
class TransformerDecoder(nn.Module):
    """
    Text decoder with cross-attention to image features.
    Uses nn.TransformerDecoder for correctness and simplicity.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        attention_dropout: float,
        max_len: int = 100,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.empty(1, max_len, hidden_dim))
        trunc_normal_(self.pos_embedding, std=0.02)

        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def _causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        # standard causal mask [sz, sz] with -inf above diagonal
        mask = torch.full((sz, sz), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        inputs: torch.Tensor,                  # [B, T] token ids
        features: Optional[torch.Tensor],      # [B, N, hidden_dim] from encoder
    ) -> torch.Tensor:
        B, T = inputs.shape
        # Embed + add positions
        x = self.embedding(inputs) + self.pos_embedding[:, :T, :]
        x = self.dropout(x)

        # Self-attention causal mask
        tgt_mask = self._causal_mask(T, x.device)

        # Cross-attention: memory = image features
        memory = features if features is not None else None

        x = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)
        x = self.norm(x)
        logits = self.lm_head(x)               # [B, T, vocab]
        return logits


# -----------------------------
# Net (API)
# -----------------------------
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        # API fields
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape

        # Channel and vocab extraction compatible with your pipeline
        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3
        # out_shape may be (V,) or ((V,), something). Safest:
        if isinstance(out_shape, (tuple, list)):
            self.vocab_size = out_shape[0] if isinstance(out_shape[0], int) else int(out_shape[0][0])
        else:
            self.vocab_size = int(out_shape)

        # Model dims (safe defaults; can be tuned by prm if present)
        self.hidden_dim = int(prm.get('hidden_dim', 768))
        self.num_heads = int(prm.get('num_heads', 8))
        self.mlp_dim = int(prm.get('mlp_dim', 3072))
        self.dropout = float(prm.get('dropout', 0.1))
        self.attn_drop = float(prm.get('attn_drop', 0.1))
        self.max_len = int(prm.get('max_len', 50))
        self.patch_size = int(prm.get('patch_size', 16))
        self.enc_layers = int(prm.get('enc_layers', 6))
        self.dec_layers = int(prm.get('dec_layers', 6))

        # Encoder (ViT)
        # Build a shape tuple like (B, C, H, W). We only need C,H,W values.
        # If in_shape already includes B, use it; otherwise assume dummy B=1.
        enc_in_shape = in_shape if len(in_shape) == 4 else (1, self.in_channels, in_shape[1], in_shape[2])
        self.encoder = VisionTransformerEncoder(
            in_shape=enc_in_shape,
            hidden_dim=self.hidden_dim,
            num_layers=self.enc_layers,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            attention_dropout=self.attn_drop,
            patch_size=self.patch_size,
        )

        # Decoder (text)
        self.decoder = TransformerDecoder(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.dec_layers,
            dropout=self.dropout,
            attention_dropout=self.attn_drop,
            max_len=self.max_len,
        )

        # Training helpers (initialized in train_setup)
        self.criterion = None
        self.optimizer = None
        self.to(self.device)

    # -------- Training API --------
    def train_setup(self, prm: dict) -> None:
        """
        Sets up optimizer & loss. Expects prm to contain 'lr' and optional 'momentum' (used as beta1).
        """
        lr = float(prm.get('lr', 1e-4))
        beta1 = float(prm.get('momentum', 0.9))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data) -> None:
        """
        One training pass over provided data loader/iterator.
        Expects batches of (images, captions) where captions are [B, T] with BOS/EOS/0-padding.
        """
        assert self.criterion is not None and self.optimizer is not None, "Call train_setup(prm) first."
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, non_blocking=True).float()
            captions = captions.to(self.device, non_blocking=True).long()

            # Teacher-forcing: predict next token
            inputs = captions[:, :-1]           # [B, T-1]
            targets = captions[:, 1:]           # [B, T-1]

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                feats = self.encoder(images)                # [B, N, H]
                logits = self.decoder(inputs, feats)        # [B, T-1, V]
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # -------- Inference/Forward --------
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # kept for API compatibility, unused
    ) -> torch.Tensor:
        """
        If captions is provided (training/eval): returns logits [B, T-1, vocab].
        If captions is None (inference): greedy decode up to max_len and return logits of the whole decoded path [B, T, vocab].
        """
        images = images.to(self.device).float()
        feats = self.encoder(images)  # [B, N, H]

        if captions is not None:
            captions = captions.to(self.device).long()
            inputs = captions[:, :-1]                         # [B, T-1]
            logits = self.decoder(inputs, feats)              # [B, T-1, V]
            return logits                                     # (tensor, not tuple) for BLEU compatibility

        # Inference (greedy)
        B = images.size(0)
        sos_id = 1  # assuming 1 is <SOS>; adapt if your tokenizer uses a different id
        cur = torch.full((B, 1), sos_id, dtype=torch.long, device=self.device)
        all_logits = []

        for _ in range(self.max_len):
            logits = self.decoder(cur, feats)                 # [B, t, V]
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # [B,1]
            all_logits.append(logits[:, -1:, :])              # keep last step logits
            cur = torch.cat([cur, next_token], dim=1)

        return torch.cat(all_logits, dim=1)                   # [B, T, V]
