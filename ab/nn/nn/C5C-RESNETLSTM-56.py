import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


class BagNetBottleneck(nn.Module):
    """
    Simple BagNet-style bottleneck:
      1x1 (reduce) -> grouped 3x3 (spatial) -> 1x1 (expand) + residual
    Optional lightweight SE after the spatial conv.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bottleneck_factor: int = 4,
        se_ratio: float | None = None,
        groups: int = 4,
    ) -> None:
        super().__init__()
        mid_channels = max(1, out_channels // bottleneck_factor)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.se = None
        if se_ratio is not None and se_ratio > 0:
            se_hidden = max(1, int(mid_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(mid_channels, se_hidden, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(se_hidden, mid_channels, kernel_size=1, bias=False),
                nn.Sigmoid(),
            )

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Residual projection if shape changes
        self.proj = (
            nn.Identity()
            if (in_channels == out_channels and stride == 1)
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))

        if self.se is not None:
            out = out * self.se(out)

        out = self.bn3(self.conv3(out))
        out = self.act(out + identity)
        return out


class Net(nn.Module):
    """
    Encoder-decoder captioning model:
      - Encoder: shallow BagNet-like CNN -> global pooled -> linear proj to 512-d
      - Decoder: Embedding + LSTM (input = [token_emb ; img_feat]) -> Linear to vocab
    """
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # Robustly extract channels and vocab size
        self.in_channels = int(in_shape[1]) if isinstance(in_shape, (tuple, list)) and len(in_shape) > 1 else int(in_shape[0])

        vocab_size = out_shape
        while isinstance(vocab_size, (tuple, list)):
            if len(vocab_size) == 0:
                raise ValueError("out_shape appears empty; cannot infer vocab size.")
            vocab_size = vocab_size[0]
        self.vocab_size = int(vocab_size)

        self.sos_index = int(prm.get("sos_idx", 1))
        self.eos_index = int(prm.get("eos_idx", 2))
        self.hidden_size = int(prm.get("hidden_size", 512))
        self.max_len = int(prm.get("max_len", 20))
        self.dropout_p = float(prm.get("dropout", 0.3))

        # ---------- Encoder ----------
        enc_channels = [64, 128, 256, 512]
        blocks = []
        in_c = self.in_channels
        for out_c in enc_channels:
            blocks.append(BagNetBottleneck(in_c, out_c, kernel_size=3, stride=2, bottleneck_factor=4, se_ratio=0.25))
            in_c = out_c
        self.encoder = nn.Sequential(*blocks)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.enc_proj = nn.Linear(enc_channels[-1], self.hidden_size)
        self.enc_act = nn.ReLU(inplace=True)

        # ---------- Decoder ----------
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size + self.hidden_size,  # token + image feature
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

        self.to(self.device)

    # ---- Utilities ----
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(images)                  # [B, C, H', W']
        feats = self.global_pool(feats).squeeze(-1).squeeze(-1)  # [B, C]
        feats = self.enc_act(self.enc_proj(feats))    # [B, hidden]
        return feats

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm["lr"]),
            betas=(float(prm.get("momentum", 0.9)), 0.999),
        )

    def learn(self, train_data):
        """Minimal training loop over (images, captions) batches."""
        self.train()
        for batch in train_data:
            if isinstance(batch, dict):
                images, captions = batch["images"], batch["captions"]
            else:
                images, captions = batch

            images = images.to(self.device)
            captions = captions.to(self.device)  # [B, T]

            # teacher forcing: input = [SOS ... T-1], target = [1 ... EOS]
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            logits, _ = self.forward(images, inputs)  # [B, T-1, V]
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

            # yield loss for potential logging
            yield loss.detach().item()

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor | None = None,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """
        If captions is provided: returns (logits, (h, c))
          logits: [B, T, V]
        Else: greedy decode up to max_len, returns (logits, (h, c))
        """
        images = images.to(self.device)
        B = images.size(0)
        img_feat = self._encode(images)  # [B, hidden]
        img_feat_seq = img_feat.unsqueeze(1)  # [B, 1, hidden]

        if captions is not None:
            # Training / teacher-forcing path
            captions = captions.to(self.device)       # [B, T]
            token_emb = self.embedding(captions)      # [B, T, hidden]
            img_rep = img_feat_seq.expand(-1, token_emb.size(1), -1)  # [B, T, hidden]
            lstm_in = torch.cat([token_emb, img_rep], dim=-1)         # [B, T, 2*hidden]

            outputs, hidden_state = self.lstm(lstm_in, hidden_state)  # [B, T, hidden]
            logits = self.fc(self.dropout(outputs))                   # [B, T, V]
            return logits, hidden_state

        # Inference path (greedy)
        tokens = torch.full((B, 1), self.sos_index, dtype=torch.long, device=self.device)
        logits_steps: list[torch.Tensor] = []
        for _ in range(self.max_len):
            token_emb = self.embedding(tokens[:, -1:])   # [B, 1, hidden]
            lstm_in = torch.cat([token_emb, img_feat_seq], dim=-1)  # [B, 1, 2*hidden]
            out, hidden_state = self.lstm(lstm_in, hidden_state)    # [B, 1, hidden]
            step_logits = self.fc(self.dropout(out))                 # [B, 1, V]
            logits_steps.append(step_logits)

            next_token = step_logits.argmax(dim=-1)  # [B, 1]
            tokens = torch.cat([tokens, next_token], dim=1)
            # Optional: stop if all hit EOS
            if (next_token == self.eos_index).all():
                break

        logits = torch.cat(logits_steps, dim=1) if logits_steps else torch.zeros(B, 0, self.vocab_size, device=self.device)
        return logits, hidden_state
