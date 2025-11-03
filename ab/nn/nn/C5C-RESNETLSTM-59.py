import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


# -------------------------
# Basic residual block
# -------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the number of channels changes, align residual with a 1x1 conv
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


# -------------------------
# Full model
# -------------------------
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        prm = prm or {}

        # Robustly infer input channels and vocab size
        def _infer_in_channels(shape):
            if isinstance(shape, (tuple, list)) and len(shape) >= 2:
                return int(shape[1])
            return 3

        def _infer_vocab_size(shape):
            x = shape
            while isinstance(x, (tuple, list)):
                if len(x) == 0:
                    raise ValueError("Invalid out_shape; cannot infer vocab size from empty tuple/list.")
                x = x[0]
            return int(x)

        self.in_channels = _infer_in_channels(in_shape)
        self.vocab_size = _infer_vocab_size(out_shape)

        # Model sizes / hyperparams
        self.hidden_size = int(prm.get("hidden_size", 768))  # >= 640
        self.num_layers = int(prm.get("num_layers", 2))
        self.max_len = int(prm.get("max_len", 20))
        self.sos_idx = int(prm.get("sos_idx", 1))
        self.eos_idx = int(prm.get("eos_idx", 2))

        # -------------------------
        # Encoder (CNN)
        # -------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 128, blocks=2)
        self.layer2 = self._make_layer(128, 256, blocks=2)
        self.layer3 = self._make_layer(256, 512, blocks=2)
        self.layer4 = self._make_layer(512, 512, blocks=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.encoder_proj = nn.Linear(512, self.hidden_size)

        # Project image features to initial LSTM hidden state
        self.h_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # -------------------------
        # Decoder (Embedding + LSTM)
        # -------------------------
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(prm.get("dropout", 0.2)))
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

        self.to(self.device)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    # -------------------------
    # Training helpers
    # -------------------------
    def train_setup(self, prm):
        self.to(self.device)
        lr = float(prm.get("lr", 1e-3))
        momentum = float(prm.get("momentum", 0.9))
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(momentum, 0.999))

    def learn(self, train_data):
        # Placeholder training loop; expects iterable of (images, captions)
        self.train()
        last_loss = 0.0
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)  # (B, T)

            # Teacher forcing: predict next tokens
            inputs = captions[:, :-1]  # (B, T-1)
            targets = captions[:, 1:]  # (B, T-1)

            logits, _ = self.forward(images, inputs)
            loss = self.criteria[0](logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
            last_loss = float(loss.detach().cpu())
        return last_loss

    # -------------------------
    # Forward (training/inference)
    # -------------------------
    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        # ----- Encoder -----
        feats = self.stem(x)
        feats = self.layer1(feats)
        feats = self.layer2(feats)
        feats = self.layer3(feats)
        feats = self.layer4(feats)
        feats = self.global_pool(feats)         # (B, 512, 1, 1)
        feats = self.flatten(feats)             # (B, 512)
        feats = self.encoder_proj(feats)        # (B, H)
        feats = self.dropout(F.relu(feats, inplace=True))

        # Init LSTM state from image embedding
        B = feats.size(0)
        h0 = torch.tanh(self.h_proj(feats)).unsqueeze(0).repeat(self.num_layers, 1, 1)  # (L, B, H)
        c0 = torch.tanh(self.c_proj(feats)).unsqueeze(0).repeat(self.num_layers, 1, 1)  # (L, B, H)
        state = (h0, c0)

        # ----- Decoder -----
        if y is not None:
            # Teacher forcing path
            emb = self.embedding(y)                  # (B, T, H)
            out, state = self.lstm(emb, state)      # (B, T, H)
            logits = self.fc(self.dropout(out))     # (B, T, V)
            return logits, state

        # Inference (greedy)
        inputs = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=x.device)
        logits_steps = []
        for _ in range(self.max_len):
            emb = self.embedding(inputs[:, -1:])    # (B, 1, H)
            out, state = self.lstm(emb, state)      # (B, 1, H)
            step_logits = self.fc(self.dropout(out))  # (B, 1, V)
            logits_steps.append(step_logits)
            next_token = step_logits.argmax(dim=-1)  # (B, 1)
            inputs = torch.cat([inputs, next_token], dim=1)
            if (next_token.squeeze(1) == self.eos_idx).all():
                break

        logits = torch.cat(logits_steps, dim=1)  # (B, L, V)
        return logits, state

    @staticmethod
    def supported_hyperparameters():
        return {"lr", "momentum"}
