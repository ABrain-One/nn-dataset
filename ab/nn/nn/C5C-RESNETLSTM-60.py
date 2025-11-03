import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def supported_hyperparameters():
    return {"lr", "momentum"}


class BottleNeck(nn.Module):
    """Lightweight bottleneck using depthwise-separable convs + residual."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, group_size: int = 64):
        super().__init__()
        mid = max(1, out_channels // 4)

        self.conv1 = self._separable_conv(in_channels, mid, kernel_size=1, stride=1, groups=min(in_channels, group_size))
        self.conv2 = self._separable_conv(mid, mid, kernel_size=kernel_size, stride=stride, groups=min(mid, group_size))
        self.conv3 = self._separable_conv(mid, out_channels, kernel_size=1, stride=1, groups=1)

        self.proj = None
        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _separable_conv(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, groups: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=in_channels if groups > 1 else 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.proj is not None:
            identity = self.proj(identity)

        out = self.relu(out + identity)
        return out


class Net(nn.Module):
    """CNN encoder + Transformer decoder (single layer) image captioning model."""
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        prm = prm or {}

        # ---- Infer channels and vocab size robustly ----
        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._infer_vocab_size(out_shape)

        # ---- Hyperparameters ----
        self.d_model = int(prm.get("hidden_size", 768))
        self.nhead = int(prm.get("num_heads", 8))
        self.encoder_depth = int(prm.get("encoder_depth", 3))
        self.decoder_dropout = float(prm.get("decoder_dropout", 0.1))
        self.max_len = int(prm.get("max_len", 20))
        self.sos_idx = int(prm.get("sos_idx", 1))
        self.eos_idx = int(prm.get("eos_idx", 2))

        # ---- Encoder ----
        stages = []
        ch = 64
        stages.append(
            nn.Sequential(
                nn.Conv2d(self.in_channels, ch, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        )
        for _ in range(self.encoder_depth):
            stages.append(BottleNeck(ch, ch * 2, kernel_size=3, stride=2))
            ch *= 2
        self.encoder_stages = nn.Sequential(*stages)

        self.enc_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.encoder_projector = nn.Linear(ch, self.d_model)

        # ---- Decoder ----
        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=2 * self.d_model,
            dropout=self.decoder_dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(self.dec_layer, num_layers=1)
        self.fc_out = nn.Linear(self.d_model, self.vocab_size)

        self.to(self.device)

    # ----------------- helpers -----------------
    @staticmethod
    def _infer_in_channels(shape: tuple) -> int:
        if isinstance(shape, (tuple, list)) and len(shape) >= 2:
            return int(shape[1])
        return 3

    @staticmethod
    def _infer_vocab_size(shape) -> int:
        x = shape
        while isinstance(x, (tuple, list)):
            if not x:
                raise ValueError("Invalid out_shape: empty tuple/list.")
            x = x[0]
        return int(x)

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        # (T, T) causal mask with True where blocked
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)
        return mask

    # ----------------- API methods -----------------
    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transformer-style: return empty tensors (not used, kept for API compatibility)."""
        return torch.empty(0, device=device), torch.empty(0, device=device)

    def train_setup(self, prm):
        self.to(self.device)
        lr = float(prm.get("lr", 1e-3))
        momentum = float(prm.get("momentum", 0.9))
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(momentum, 0.999))

    def learn(self, train_data):
        """Simple training loop expecting an iterable of (images, captions)."""
        self.train()
        last_loss = 0.0
        for images, captions in train_data:
            images = images.to(self.device)                # (B, C, H, W)
            captions = captions.to(self.device)            # (B, T)

            # teacher forcing: predict t+1 from up-to-t
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            logits, _ = self.forward(images, inputs)
            loss = self.criteria[0](logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
            last_loss = float(loss.detach().cpu())
        return last_loss

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None):
        # ----- Encode image to memory tokens (B, 1, D) -----
        feats = self.encoder_stages(images)            # (B, ch, H', W')
        feats = self.enc_pool(feats).squeeze(-1).squeeze(-1)  # (B, ch)
        memory = self.encoder_projector(feats).unsqueeze(1)    # (B, 1, D)

        if captions is not None:
            # ----- Training: run decoder with causal mask -----
            tgt = self.embedding(captions)  # (B, T, D)
            T = tgt.size(1)
            tgt_mask = self._generate_square_subsequent_mask(T, tgt.device)  # (T, T)
            out = self.decoder(tgt, memory, tgt_mask=tgt_mask)               # (B, T, D)
            logits = self.fc_out(out)                                        # (B, T, V)
            return logits, None

        # ----- Inference: greedy decode -----
        B = images.size(0)
        generated = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=images.device)
        all_logits = []
        for _ in range(self.max_len):
            tgt = self.embedding(generated)                    # (B, T, D)
            T = tgt.size(1)
            tgt_mask = self._generate_square_subsequent_mask(T, tgt.device)
            out = self.decoder(tgt, memory, tgt_mask=tgt_mask) # (B, T, D)
            step_logits = self.fc_out(out[:, -1:, :])          # (B, 1, V)
            all_logits.append(step_logits)
            next_token = step_logits.argmax(-1)                # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token.squeeze(1) == self.eos_idx).all():
                break
        if not all_logits:
            # No steps taken; produce a single step from SOS to avoid empty tensors
            out0 = self.decoder(self.embedding(generated), memory, tgt_mask=self._generate_square_subsequent_mask(generated.size(1), images.device))
            all_logits = [self.fc_out(out0[:, -1:, :])]
        logits = torch.cat(all_logits, dim=1)                  # (B, L, V)
        return logits, None
