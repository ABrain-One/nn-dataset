import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple


def supported_hyperparameters():
    return {"lr", "momentum"}


# --------------------- Building Blocks ---------------------
class SEAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        reduced = max(1, channels // ratio)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = x.mean(dim=(2, 3))          # [B, C]
        y = self.relu(self.fc1(y))      # [B, C//r]
        y = self.sigmoid(self.fc2(y))   # [B, C]
        y = y.view(b, c, 1, 1)          # [B, C, 1, 1]
        return x * y


class CBAM(nn.Module):
    def __init__(self, channels: int, spatial_kernel: int = 7):
        super().__init__()
        self.ca = SEAttention(channels)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ca = self.ca(x)
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        x_sp = torch.cat([avg_out, max_out], dim=1)     # [B, 2, H, W]
        sp_weight = self.spatial(x_sp)                  # [B, 1, H, W]
        return x_ca * sp_weight


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
            nn.BatchNorm2d(c_in),
            nn.SiLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


# --------------------- Encoder / Decoder ---------------------
class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        ch1, ch2, ch3, ch4 = 64, 128, 256, 512
        self.net = nn.Sequential(
            DepthwiseSeparableConv(in_channels, ch1, stride=1),
            CBAM(ch1),
            DepthwiseSeparableConv(ch1, ch2, stride=2),
            CBAM(ch2),
            DepthwiseSeparableConv(ch2, ch3, stride=2),
            CBAM(ch3),
            DepthwiseSeparableConv(ch3, ch4, stride=2),
            CBAM(ch4),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Conv2d(ch4, hidden_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)         # [B, 512, 1, 1]
        x = self.proj(x)        # [B, H, 1, 1]
        return x.flatten(1)     # [B, H]


class Decoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.mem_proj = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(1)
        emb = self.embed(inputs)                 # [B, T, H]
        if memory is not None:
            mem_add = self.mem_proj(memory).unsqueeze(1)  # [B,1,H]
            emb = emb + mem_add

        out, h_n = self.gru(emb, hidden_state)  # [B, T, H]
        logits = self.fc(out)                   # [B, T, V]
        return logits, h_n


# --------------------- Top-level Net ---------------------
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.prm = prm or {}

        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)
        self.hidden_size = int(self.prm.get("hidden_size", 768))

        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        self.encoder = Encoder(self.in_channels, self.hidden_size)
        self.decoder = Decoder(self.hidden_size, self.vocab_size)

        self.sos_idx = int(self.prm.get("sos", self.prm.get("sos_idx", 1)))
        self.eos_idx = int(self.prm.get("eos", 0))
        self.pad_idx = int(self.prm.get("pad_idx", 0))
        self.max_len = int(self.prm.get("max_length", 20))

        self.criterion: Optional[nn.Module] = None
        self.criteria = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    @staticmethod
    def supported_hyperparameters():
        return {"lr", "momentum"}

    # ---- Training API ----
    def train_setup(self, prm: dict):
        prm = prm or {}
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device)
        self.criteria = (self.criterion,)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data: Any):
        self.train()
        if self.optimizer is None or self.criterion is None:
            self.train_setup(getattr(train_data, "prm", self.prm))

        for batch in train_data:
            if isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    continue
                images, captions = batch[0], batch[1]
            elif isinstance(batch, dict):
                images = batch.get("x", None)
                captions = batch.get("y", None)
                if images is None or captions is None:
                    continue
            else:
                images = getattr(batch, "x", None)
                captions = getattr(batch, "y", None)
                if images is None or captions is None:
                    continue

            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device).long()
            if captions.dim() == 3 and captions.size(1) == 1:
                captions = captions[:, 0, :]
            if captions.size(1) <= 1:
                continue

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            self.optimizer.zero_grad(set_to_none=True)
            logits, _ = self.forward(images, inputs)  # [B, T-1, V]
            loss = self.criterion(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # ---- Forward / Inference ----
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)  # [B, H]

        if hidden_state is None:
            hidden_state = self.decoder.init_hidden(images.size(0), self.device)

        if captions is not None:
            logits, h_n = self.decoder(captions.to(self.device).long(), hidden_state, memory)
            return logits, h_n

        bsz = images.size(0)
        h = hidden_state
        seq = torch.full((bsz, 1), self.sos_idx, dtype=torch.long, device=self.device)

        for _ in range(self.max_len):
            logits, h = self.decoder(seq[:, -1:], h, memory)  # last token only
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=1)
            if (next_tok == self.eos_idx).all():
                break

        return seq, h

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        h = self.decoder.init_hidden(images.size(0), self.device)

        seq = torch.full((images.size(0), 1), self.sos_idx, dtype=torch.long, device=self.device)
        for _ in range(self.max_len):
            logits, h = self.decoder(seq[:, -1:], h, memory)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=1)
            if (next_tok == self.eos_idx).all():
                break
        return seq[:, 1:]

    # ---- Utils ----
    @staticmethod
    def _first_int(x) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            for item in x:
                try:
                    return Net._first_int(item)
                except Exception:
                    continue
        return int(x)

    @staticmethod
    def _infer_in_channels(in_shape: Any) -> int:
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) == 3 and isinstance(in_shape[0], int):
                return int(in_shape[0])          # (C,H,W)
            if len(in_shape) >= 2 and isinstance(in_shape[1], int):
                return int(in_shape[1])          # (B,C,H,W)
        return 3
