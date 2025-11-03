import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Iterable


def supported_hyperparameters():
    return {"lr", "momentum"}


# -------------------------
# ResNet-style basic block
# -------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


# -------------------------
# Encoder (ResNet-ish) -> [B, 1, H]
# -------------------------
class EncoderCNN(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, hidden_size)

        self._init_weights()

    @staticmethod
    def _make_layer(inplanes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(inplanes, planes, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)  # [B, 512]
        x = self.proj(x)                # [B, H]
        return x.unsqueeze(1)           # [B, 1, H]


# -------------------------
# Decoder (LSTM) -> logits
# -------------------------
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 1, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.pad_idx = int(pad_idx)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

    def _normalize_hidden(
        self, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]], batch: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=device)
            c0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=device)
            return h0, c0
        h, c = hidden
        # ensure shape [num_layers, B, H]
        if h.dim() == 2:
            h = h.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        if c.dim() == 2:
            c = c.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        return h, c

    def forward(
        self,
        tokens: torch.Tensor,                              # [B, T]
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]],
        *,
        features: torch.Tensor                             # [B, 1, H]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T = tokens.shape
        device = tokens.device

        emb = self.embedding(tokens)                       # [B, T, H]
        # add image context at every step
        if features.dim() == 2:
            features = features.unsqueeze(1)               # [B,1,H]
        img_ctx = features.expand(B, T, features.size(-1)) # [B,T,H]
        x = self.dropout(emb + img_ctx)

        h0, c0 = self._normalize_hidden(hidden, B, device)
        out, (hn, cn) = self.lstm(x, (h0, c0))             # [B,T,H]
        logits = self.fc_out(self.dropout(out))            # [B,T,V]
        return logits, (hn, cn)

    @torch.no_grad()
    def greedy_decode(
        self,
        features: torch.Tensor,                            # [B, 1, H]
        *,
        max_len: int,
        sos_idx: int,
        eos_idx: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = features.size(0)
        tokens = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)  # [B,1]
        hidden = None
        all_logits = []

        for _ in range(max_len):
            step_logits, hidden = self.forward(tokens[:, -1:], hidden, features=features)  # [B,1,V]
            all_logits.append(step_logits)
            next_tok = step_logits.argmax(-1)  # [B,1]
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok.squeeze(1) == eos_idx).all():
                break

        return tokens, torch.cat(all_logits, dim=1) if all_logits else torch.zeros(B, 0, self.vocab_size, device=device)


# -------------------------
# Main Net
# -------------------------
class Net(nn.Module):
    """
    Image Captioning Net (encoder + decoder).
    Exposes:
      - self.cnn : EncoderCNN
      - self.rnn : DecoderRNN
      - train_setup(prm), learn(train_data), forward(images, captions|None)
    """
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device

        # infer channels / vocab
        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # hyperparams
        self.hidden_size = int(prm.get("hidden_size", 512))
        self.num_layers = int(prm.get("num_layers", 1))
        self.dropout = float(prm.get("dropout", 0.1))
        self.pad_idx = int(prm.get("pad_idx", 0))
        self.sos_idx = int(prm.get("sos_idx", 1))
        self.eos_idx = int(prm.get("eos_idx", 2))
        self.max_len = int(prm.get("max_length", prm.get("max_len", 20)))

        # modules
        self.cnn = EncoderCNN(self.in_channels, hidden_size=self.hidden_size)         # -> [B,1,H]
        self.rnn = DecoderRNN(self.vocab_size, hidden_size=self.hidden_size,
                              num_layers=self.num_layers, dropout=self.dropout, pad_idx=self.pad_idx)

        self.criteria: Tuple[nn.Module, ...] = ()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.to(self.device)

    @staticmethod
    def _infer_in_channels(in_shape) -> int:
        if isinstance(in_shape, (tuple, list)):
            # often (B,C,H,W) or (C,H,W)
            if len(in_shape) == 4:
                return int(in_shape[1])
            if len(in_shape) == 3:
                return int(in_shape[0])
        return 3

    @staticmethod
    def _first_int(x) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return Net._first_int(x[0])
        raise ValueError(f"Cannot infer integer from out_shape={x}")

    # ---- training helpers ----
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm["lr"], betas=(prm.get("momentum", 0.9), 0.999)
        )

    def learn(self, train_data: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
        if not self.criteria or self.optimizer is None:
            return

        self.train()
        for images, captions in train_data:
            images = images.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)

            if captions.dim() == 3:        # [B,1,T] -> [B,T]
                captions = captions[:, 0, :]

            if captions.size(1) <= 1:
                continue

            inputs = captions[:, :-1]      # teacher forcing input
            targets = captions[:, 1:]      # predict next tokens

            features = self.cnn(images)                         # [B,1,H]
            logits, _ = self.rnn(inputs, None, features=features)  # [B,T-1,V]

            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # ---- forward ----
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        With captions (teacher forcing):
            returns (logits [B,T-1,V], hidden_state)
        Without captions (inference, greedy):
            returns (logits [B,T_gen,V], (None,None))
        """
        images = images.to(self.device)
        features = self.cnn(images)  # [B,1,H]

        # teacher forcing
        if captions is not None:
            captions = captions.to(self.device)
            if captions.dim() == 3:
                captions = captions[:, 0, :]
            if captions.size(1) <= 1:
                empty = torch.zeros(captions.size(0), 0, self.vocab_size, device=self.device)
                return empty, hidden_state
            inputs = captions[:, :-1]
            logits, hidden_state = self.rnn(inputs, hidden_state, features=features)
            return logits, hidden_state

        # greedy inference
        tokens, gen_logits = self.rnn.greedy_decode(
            features, max_len=self.max_len, sos_idx=self.sos_idx, eos_idx=self.eos_idx, device=self.device
        )
        return gen_logits, (None, None)


# --------------- self-check ---------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 3, 224, 224
    V = 5000
    in_shape = (B, C, H, W)
    out_shape = (V,)
    prm = dict(lr=1e-4, momentum=0.9, hidden_size=512, num_layers=1, dropout=0.1,
               pad_idx=0, sos_idx=1, eos_idx=2, max_length=16)

    net = Net(in_shape, out_shape, prm, device).to(device)
    net.train_setup(prm)

    # teacher forcing path
    imgs = torch.randn(B, C, H, W, device=device)
    caps = torch.randint(0, V, (B, 12), device=device)
    caps[:, 0] = prm["sos_idx"]
    logits, _ = net(imgs, caps)        # [B,T-1,V]
    print("Teacher-forcing logits:", tuple(logits.shape))

    # inference path
    gen_logits, _ = net(imgs, captions=None)
    print("Greedy logits:", tuple(gen_logits.shape))
