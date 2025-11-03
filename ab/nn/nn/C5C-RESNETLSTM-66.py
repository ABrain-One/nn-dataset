import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

# ---------- Hyperparameters ----------
def supported_hyperparameters():
    return {"lr", "momentum"}


# ---------- Encoder/Decoder building blocks ----------

class SE(nn.Module):
    """Squeeze-and-Excitation block (optional channel attention)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.act(self.fc1(y))
        y = self.gate(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class EncoderCNN(nn.Module):
    """Lightweight CNN encoder that produces a single feature vector per image."""
    def __init__(self, in_channels: int, hidden_size: int = 768, use_se: bool = True):
        super().__init__()
        ch = [64, 128, 256, 512]  # stages

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        def block(cin, cout, stride: int):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                SE(cout, reduction=16) if use_se else nn.Identity(),
                nn.ReLU(inplace=True),
            )

        self.stage1 = block(ch[0], ch[1], stride=2)
        self.stage2 = block(ch[1], ch[2], stride=2)
        self.stage3 = block(ch[2], ch[3], stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(ch[3], hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)      # [B, 512]
        x = self.proj(x)                 # [B, hidden]
        return x


class DecoderRNN(nn.Module):
    """Simple LSTM decoder with learned token embeddings."""
    def __init__(self, vocab_size: int, hidden_size: int = 768, num_layers: int = 1,
                 dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        # Dropout inside LSTM is only applied if num_layers > 1
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        # Projections to initialize hidden state from image feature
        self.init_h = nn.Linear(hidden_size, hidden_size)
        self.init_c = nn.Linear(hidden_size, hidden_size)

    def init_state(self, img_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        img_feat: [B, hidden]
        returns (h0, c0) each of shape [num_layers, B, hidden]
        """
        h0 = torch.tanh(self.init_h(img_feat))
        c0 = torch.tanh(self.init_c(img_feat))
        # Repeat across layers
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()
        c0 = c0.unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()
        return h0, c0

    def forward(self, input_ids: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        input_ids: [B, T] (token ids)
        returns logits: [B, T, V]
        """
        emb = self.embedding(input_ids)         # [B, T, H]
        out, hidden = self.lstm(emb, hidden)    # [B, T, H]
        logits = self.fc(out)                   # [B, T, V]
        return logits, hidden


# ---------- Main model ----------

class Net(nn.Module):
    """
    Image Captioning model with CNN encoder + LSTM decoder.

    API preserved:
      - __init__(in_shape, out_shape, prm, device)
      - train_setup(prm)
      - learn(train_data)
      - forward(images, captions=None, hidden_state=None)
      - init_zero_hidden(batch, device)
    """
    def __init__(self, in_shape: Tuple, out_shape: Tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm or {}

        # Infer channels/height/width from in_shape
        # Accept (C,H,W) or (B,C,H,W)
        if len(in_shape) == 4:
            in_channels = int(in_shape[1])
        elif len(in_shape) == 3:
            in_channels = int(in_shape[0])
        else:
            raise ValueError(f"in_shape should be (C,H,W) or (B,C,H,W), got {in_shape}")

        # Robustly infer vocab size (supports nested tuples like seen in prior files)
        self.vocab_size = int(self._first_int(out_shape))
        if self.vocab_size <= 0:
            raise ValueError(f"Invalid out_shape {out_shape}: could not infer positive vocab size")

        # Hyperparameters with defaults
        self.hidden_size = int(self.prm.get("hidden_size", 768))
        self.num_layers = int(self.prm.get("num_layers", 1))
        self.dropout = float(self.prm.get("dropout", 0.1))
        self.pad_idx = int(self.prm.get("pad_idx", 0))
        self.sos_idx = int(self.prm.get("sos_idx", 1))
        self.eos_idx = int(self.prm.get("eos_idx", 2))
        self.max_len = int(self.prm.get("max_len", 20))

        # Modules
        self.encoder = EncoderCNN(in_channels=in_channels, hidden_size=self.hidden_size, use_se=True)
        self.decoder = DecoderRNN(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            pad_idx=self.pad_idx,
        )

        # Training helpers (populated in train_setup)
        self.criteria: Tuple[nn.Module, ...] = ()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.grad_clip = float(self.prm.get("grad_clip", 3.0))

        # Move to device
        self.to(self.device)

    # ---------- Training API ----------

    def train_setup(self, prm: dict):
        """Configure loss and optimizer."""
        prm = prm or {}
        lr = float(prm.get("lr", 1e-4))
        beta1 = float(prm.get("momentum", 0.9))

        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        """
        Simple training loop.
        Expects an iterable of (images, captions) where:
          images  : [B, C, H, W]
          captions: [B, T] (with SOS at position 0 and PAD=0)
        """
        if self.optimizer is None or not self.criteria:
            raise RuntimeError("Call train_setup(prm) before learn().")

        self.train()
        for images, captions in train_data:
            images = images.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)

            # Teacher forcing: predict next tokens for all but the last position
            inputs = captions[:, :-1]          # [B, T-1]
            targets = captions[:, 1:]          # [B, T-1]

            logits, _ = self.forward(images, inputs)  # [B, T-1, V]
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            self.optimizer.step()

            # Optional: yield or print for progress tracking
            # yield float(loss.item())

    # ---------- Inference / Forward ----------

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        If captions is provided (teacher forcing):
            images: [B, C, H, W]
            captions: [B, T] of token ids
            returns (logits [B,T,V], hidden_state)

        If captions is None (greedy decode):
            returns (logits [B,T,V], hidden_state) for generated sequence
        """
        # Encode images to a single vector per image
        img_feat = self.encoder(images)  # [B, H]
        if hidden_state is None:
            hidden_state = self.decoder.init_state(img_feat)

        if captions is not None:
            # Teacher forcing path
            logits, hidden_state = self.decoder(captions, hidden_state)  # [B,T,V]
            return logits, hidden_state

        # Inference path (greedy)
        B = images.size(0)
        inputs = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)  # [B,1]
        logits_steps = []
        ended = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(self.max_len):
            step_logits, hidden_state = self.decoder(inputs[:, -1:], hidden_state)  # only last token
            # step_logits: [B, 1, V]
            logits_steps.append(step_logits)
            next_token = step_logits.argmax(dim=-1)  # [B,1]
            inputs = torch.cat([inputs, next_token], dim=1)
            ended |= (next_token.squeeze(1) == self.eos_idx)
            if ended.all():
                break

        logits = torch.cat(logits_steps, dim=1)  # [B, T_gen, V]
        return logits, hidden_state

    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero state (not used in normal flow since we init from image features)."""
        h0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=device)
        return h0, c0

    # ---------- Utilities ----------

    @staticmethod
    def _first_int(x: Union[int, Tuple, list]) -> int:
        """Extract the first integer from possibly nested tuples/lists."""
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return Net._first_int(x[0])
        raise ValueError(f"Cannot infer integer from {x}")


# ---------- Minimal self-test (runs on CPU/GPU) ----------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Shapes
    B, C, H, W = 2, 3, 224, 224
    vocab_size = 1000

    # Model
    in_shape = (B, C, H, W)  # accepts (B,C,H,W) or (C,H,W)
    out_shape = (vocab_size,)  # robustly parsed by _first_int
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prm = {"lr": 1e-4, "momentum": 0.9, "hidden_size": 768, "num_layers": 1, "dropout": 0.1}

    model = Net(in_shape, out_shape, prm, device)
    model.train_setup(prm)

    # Dummy batch
    images = torch.randn(B, C, H, W, device=device)
    captions = torch.randint(0, vocab_size, (B, 12), device=device)
    captions[:, 0] = 1  # SOS
    captions[:, -1] = 2  # EOS

    # Forward (teacher forcing)
    logits, _ = model(images, captions[:, :-1])
    print("Teacher-forcing logits:", tuple(logits.shape))  # [B, T-1, V]

    # Greedy decode
    gen_logits, _ = model(images, captions=None)
    print("Generated logits:", tuple(gen_logits.shape))    # [B, T_gen, V]
