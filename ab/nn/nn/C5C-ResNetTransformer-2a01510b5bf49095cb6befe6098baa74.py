import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    # Kept minimal on purpose to fit the NN-Dataset API expectations
    return {'lr', 'momentum'}


# -------------------------- Encoder (ResNet-ish) --------------------------

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.avg(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
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


class Encoder(nn.Module):
    """
    Produces a single 'memory' token per image of size d_model.
    Output shape: [B, 1, d_model]
    """
    def __init__(self, in_channels: int, d_model: int = 512):
        super().__init__()
        c1, c2, c3 = 64, 128, 256
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 7, 2, 3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = BasicBlock(c1, c1, stride=1, use_se=True)
        self.layer2 = BasicBlock(c1, c2, stride=2, use_se=True)
        self.layer3 = BasicBlock(c2, c3, stride=2, use_se=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(c3, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)                      # [B, d_model]
        return x.unsqueeze(1)                 # [B, 1, d_model]


# -------------------------- Transformer Decoder --------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0: d_model: 2] = torch.sin(position * div_term)
        pe[:, 1: d_model: 2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, num_layers: int = 2, dim_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, T: int, device: torch.device):
        # upper triangular -inf above diagonal
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def forward(self, tgt_tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        tgt_tokens: [B, T] (teacher-forced tokens without the last token)
        memory: [B, S=1, d_model]
        returns logits: [B, T, vocab_size]
        """
        x = self.embedding(tgt_tokens) * math.sqrt(self.d_model)
        x = self.pos(x)
        T = x.size(1)
        mask = self._causal_mask(T, x.device)
        out = self.decoder(tgt=x, memory=memory, tgt_mask=mask)
        return self.out(out)

    @torch.no_grad()
    def greedy_decode(self, memory: torch.Tensor, max_len: int = 50, start_id: int = 1, end_id: int = 2) -> torch.Tensor:
        """
        memory: [B, 1, d_model]
        returns tokens: [B, <=max_len]
        """
        B = memory.size(0)
        ys = torch.full((B, 1), start_id, dtype=torch.long, device=memory.device)
        for _ in range(max_len):
            x = self.embedding(ys) * math.sqrt(self.d_model)
            x = self.pos(x)
            T = x.size(1)
            mask = self._causal_mask(T, x.device)
            out = self.decoder(tgt=x, memory=memory, tgt_mask=mask)            # [B, T, d_model]
            next_logits = self.out(out[:, -1, :])                               # [B, V]
            next_ids = next_logits.argmax(dim=-1, keepdim=True)                # [B, 1]
            ys = torch.cat([ys, next_ids], dim=1)
            if (next_ids.squeeze(1) == end_id).all():
                break
        return ys[:, 1:]  # discard SOS


# -------------------------- Top-level Net --------------------------

class Net(nn.Module):
    """
    Image Captioning: CNN encoder + Transformer decoder.

    Required API:
      - __init__(in_shape, out_shape, prm, device)
      - train_setup(prm)
      - learn(train_data)
      - forward(images, captions=None, hidden_state=None)
    """
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        in_channels = int(in_shape[1])
        vocab_size = int(out_shape[0])

        d_model = int(prm.get('hidden_dim', 512))
        nhead = 8 if d_model % 8 == 0 else 4
        num_layers = int(prm.get('num_layers', 2))

        self.encoder = Encoder(in_channels, d_model=d_model)
        self.decoder = TransformerDecoder(vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.vocab_size = vocab_size

        # default training components (configured in train_setup)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = None

    # --- helpers ---
    @staticmethod
    def _norm_caps(captions: torch.Tensor | None) -> torch.Tensor | None:
        if captions is None:
            return None
        if captions.ndim == 3:
            captions = captions[:, 0, :]
        elif captions.ndim == 1:
            captions = captions.unsqueeze(0)
        return captions.long()

    # --- public API ---
    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = float(prm.get('lr', 1e-3))
        momentum = float(prm.get('momentum', 0.9))
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        self.criterion = self.criterion.to(self.device)

    def learn(self, train_data):
        self.train()
        for images, caps in train_data:
            images = images.to(self.device)
            caps = caps.to(self.device)
            caps = self._norm_caps(caps)                           # [B, T]
            inputs = caps[:, :-1]                                  # [B, T-1]
            targets = caps[:, 1:]                                  # [B, T-1]
            memory = self.encoder(images)                          # [B, 1, D]
            logits = self.decoder(inputs, memory)                  # [B, T-1, V]
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: torch.Tensor | None = None, hidden_state=None):
        memory = self.encoder(images.to(self.device))              # [B, 1, D]
        if captions is not None:
            caps = self._norm_caps(captions)                       # [B, T]
            inputs = caps[:, :-1]
            return self.decoder(inputs, memory)                    # logits [B, T-1, V]
        # inference
        return self.decoder.greedy_decode(memory, max_len=50, start_id=1, end_id=2)
