import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple


def supported_hyperparameters():
    return {"lr", "momentum"}


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (safe 4D implementation)."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)      # [B, C, 1, 1]
        y = self.fc(y)            # [B, C, 1, 1]
        return x * y              # broadcast multiply


class Net(nn.Module):
    """
    Image encoder (small CNN) + Transformer decoder captioner.
    Cleaned up to compile and run with teacher forcing or greedy decode.
    """

    def __init__(
        self, in_shape: tuple, out_shape: tuple, prm: Dict[str, Any], device: torch.device
    ) -> None:
        super().__init__()
        self.device = device
        self.prm = prm or {}

        # ---- Shapes / sizes ----
        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._infer_vocab_size(out_shape)

        # ---- Core hyperparams ----
        self.hidden_size = int(self.prm.get("hidden_size", 768))
        self.nheads = int(self.prm.get("nheads", 12))
        self.num_layers = int(self.prm.get("num_layers", 6))
        self.max_len = int(self.prm.get("max_len", 50))
        self.sos_idx = int(self.prm.get("sos_idx", 1))
        self.eos_idx = int(self.prm.get("eos_idx", 2))
        self.dropout = float(self.prm.get("dropout", 0.3))

        # ---- Encoder (backbone) ----
        self.backbone_encoder = nn.Sequential(
            BasicConv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            BasicConv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicConv2d(128, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            SEBlock(256),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Linear(256, self.hidden_size)

        # ---- Decoder (Transformer) ----
        self.token_embed = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.pos_embed = nn.Embedding(self.max_len, self.hidden_size)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.nheads,
            dim_feedforward=self.hidden_size * 4,
            dropout=min(self.dropout, 0.3),
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=dec_layer, num_layers=self.num_layers, norm=nn.LayerNorm(self.hidden_size)
        )
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

        self.to(self.device)

    # ---------- helpers ----------
    @staticmethod
    def _infer_in_channels(in_shape) -> int:
        # Expect (B, C, H, W) or (C, H, W) style; fall back to 3
        try:
            return int(in_shape[1])
        except Exception:
            return 3

    @staticmethod
    def _infer_vocab_size(out_shape) -> int:
        x = out_shape
        while isinstance(x, (tuple, list)):
            if len(x) == 0:
                raise ValueError("Invalid out_shape: empty container")
            x = x[0]
        return int(x)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        # (L, L) mask with True where future positions are masked
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    # ---------- API ----------
    def train_setup(self, prm: Dict[str, Any]):
        self.to(self.device)
        lr = float(prm.get("lr", 1e-4))
        beta1 = float(prm.get("momentum", 0.9))
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))
        self.grad_clip_value = 3.0

    def learn(self, train_data):
        """Teacher-forcing loop over an iterable of (images, captions)."""
        self.train()
        last_loss = 0.0
        for images, captions in train_data:
            images = images.to(self.device)                  # (B, C, H, W)
            captions = captions.to(self.device)              # (B, T) or (B, 1, T)
            if captions.dim() == 3:
                captions = captions[:, 0, :]

            inp = captions[:, :-1]                           # (B, T-1)
            tgt = captions[:, 1:]                            # (B, T-1)

            logits, _ = self.forward(images, inp)            # (B, T-1, V)
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), tgt.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_value)
            self.optimizer.step()
            last_loss = float(loss.detach().cpu())
        return last_loss

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """
        If captions is provided: teacher forcing; returns (logits, None).
        If captions is None: greedy decoding; returns (logits, None) over generated steps.
        """
        B = images.size(0)

        # ---- Encode image to a single token memory ----
        feats = self.backbone_encoder(images)           # [B, 256, H', W']
        pooled = self.adaptive_pool(feats).view(B, 256) # [B, 256]
        memory = self.projector(pooled).unsqueeze(1)    # [B, 1, hidden]

        if captions is not None:
            # Teacher forcing: embed input tokens (no last token)
            T = captions.size(1)
            positions = torch.arange(T, device=images.device).unsqueeze(0).expand(B, T)
            tgt_embed = self.token_embed(captions) + self.pos_embed(positions)  # [B, T, hidden]

            tgt_mask = self._causal_mask(T, images.device)  # (T, T)
            out = self.transformer_decoder(
                tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask, memory_key_padding_mask=None
            )  # [B, T, hidden]
            logits = self.fc_out(out)  # [B, T, V]
            return logits, None

        # Greedy decoding
        tokens = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=images.device)
        logits_steps = []
        for step in range(self.max_len):
            T = tokens.size(1)
            positions = torch.arange(T, device=images.device).unsqueeze(0).expand(B, T)
            tgt_embed = self.token_embed(tokens) + self.pos_embed(positions)  # [B, T, hidden]
            tgt_mask = self._causal_mask(T, images.device)
            out = self.transformer_decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask)
            step_logits = self.fc_out(out[:, -1:])  # [B, 1, V]
            logits_steps.append(step_logits)
            next_tok = step_logits.argmax(-1)       # [B, 1]
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == self.eos_idx).all():
                break

        logits = torch.cat(logits_steps, dim=1) if logits_steps else torch.empty(B, 0, self.vocab_size, device=images.device)
        return logits, None
