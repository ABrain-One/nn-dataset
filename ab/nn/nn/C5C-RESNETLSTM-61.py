import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def supported_hyperparameters():
    return {"lr", "momentum"}


class SqueezeExcitation(nn.Module):
    """Standard S/E block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x))
        return x * scale


class ResidualBlock(nn.Module):
    """Simple residual block with S/E."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SqueezeExcitation(out_ch, reduction=16)
        self.relu = nn.ReLU(inplace=True)

        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
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


class Net(nn.Module):
    """
    CNN encoder -> Transformer decoder captioning model.
    Clean, minimal, and compiles.
    """
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        prm = prm or {}

        # Infer shapes robustly
        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._infer_vocab_size(out_shape)

        # Hyperparameters
        self.d_model = int(prm.get("hidden_size", 768))
        self.nhead = int(prm.get("num_heads", 8))
        self.max_len = int(prm.get("max_len", 20))
        self.sos_idx = int(prm.get("sos_idx", 1))
        self.eos_idx = int(prm.get("eos_idx", 2))

        # ---- Encoder ----
        e_ch = 64
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, e_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(e_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.enc_layers = nn.Sequential(
            ResidualBlock(e_ch, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
        )
        self.enc_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.enc_proj = nn.Linear(512, self.d_model)

        # ---- Decoder ----
        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=4 * self.d_model,
            dropout=float(prm.get("decoder_dropout", 0.1)),
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(prm.get("decoder_layers", 2)))
        self.fc_out = nn.Linear(self.d_model, self.vocab_size)

        self.to(self.device)

    # --------- utilities ----------
    @staticmethod
    def _infer_in_channels(shape) -> int:
        try:
            return int(shape[1])
        except Exception:
            return 3

    @staticmethod
    def _infer_vocab_size(shape) -> int:
        x = shape
        while isinstance(x, (tuple, list)):
            if len(x) == 0:
                raise ValueError("Invalid out_shape (empty).")
            x = x[0]
        return int(x)

    @staticmethod
    def _causal_mask(sz: int, device) -> torch.Tensor:
        # (T, T) with True where future tokens are masked
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)

    # --------- API methods ----------
    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Transformer-style: return empty tensors for API compatibility
        return torch.empty(0, device=device), torch.empty(0, device=device)

    def train_setup(self, prm):
        self.to(self.device)
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        self.train()
        last_loss = 0.0
        for images, captions in train_data:
            images = images.to(self.device)          # (B, C, H, W)
            captions = captions.to(self.device)      # (B, T)

            # Teacher forcing: predict next token
            inp = captions[:, :-1]                   # (B, T-1)
            tgt = captions[:, 1:]                    # (B, T-1)

            logits, _ = self.forward(images, inp)    # (B, T-1, V)
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), tgt.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
            last_loss = float(loss.detach().cpu())
        return last_loss

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # ---- Encode image ----
        x = self.stem(images)                     # (B, 64, H', W')
        x = self.enc_layers(x)                    # (B, 512, H'', W'')
        x = self.enc_pool(x).flatten(1)           # (B, 512)
        memory = self.enc_proj(x).unsqueeze(1)    # (B, 1, D)

        if captions is not None:
            # ---- Training path ----
            tgt = self.embedding(captions)        # (B, T, D)
            T = tgt.size(1)
            mask = self._causal_mask(T, tgt.device)
            out = self.decoder(tgt=tgt, memory=memory, tgt_mask=mask)  # (B, T, D)
            logits = self.fc_out(out)             # (B, T, V)
            return logits, None

        # ---- Inference (greedy) ----
        B = images.size(0)
        generated = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=images.device)
        logits_collect = []

        for _ in range(self.max_len):
            tgt = self.embedding(generated)                   # (B, T, D)
            T = tgt.size(1)
            mask = self._causal_mask(T, tgt.device)
            out = self.decoder(tgt=tgt, memory=memory, tgt_mask=mask)  # (B, T, D)
            step_logits = self.fc_out(out[:, -1:, :])         # (B, 1, V)
            logits_collect.append(step_logits)
            next_tok = step_logits.argmax(-1)                 # (B, 1)
            generated = torch.cat([generated, next_tok], dim=1)
            if (next_tok.squeeze(1) == self.eos_idx).all():
                break

        if not logits_collect:
            out0 = self.decoder(self.embedding(generated), memory, tgt_mask=self._causal_mask(generated.size(1), images.device))
            logits_collect = [self.fc_out(out0[:, -1:, :])]

        logits = torch.cat(logits_collect, dim=1)             # (B, L, V)
        return logits, None
