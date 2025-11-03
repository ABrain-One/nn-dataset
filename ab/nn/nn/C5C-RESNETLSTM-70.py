import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Iterable, Union


def supported_hyperparameters():
    return {"lr", "momentum"}


# -------------------------
# Encoder: CNN -> tokens
# -------------------------
class CNNEncoderTokens(nn.Module):
    """
    Lightweight CNN encoder that produces a single token per image.
    Output shape: [B, 1, hidden_size]
    """
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)                  # [B, 256, H', W']
        x = self.pool(x).flatten(1)           # [B, 256]
        x = self.proj(x)                      # [B, H]
        return x.unsqueeze(1)                 # [B, 1, H]


# -------------------------
# Decoder: LSTM (+ image ctx)
# -------------------------
class DecoderRNN(nn.Module):
    """
    LSTM decoder with word embeddings. Adds the encoded image token to each step.
    API matches how Net.forward uses self.rnn(tokens, hidden, features=features).
    """
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.pad_idx = int(pad_idx)

        self.embed = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def _normalize_hidden(
        self, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]], batch: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=device)
            c0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=device)
            return h0, c0
        # Accept (B,H) or (num_layers,B,H)
        h, c = hidden
        if h.dim() == 2:  # [B,H]
            h = h.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        if c.dim() == 2:  # [B,H]
            c = c.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        return h, c

    def forward(
        self,
        tokens: torch.Tensor,                              # [B, T] teacher-forcing tokens
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]],
        *,
        features: torch.Tensor                             # [B, 1, H] image token(s)
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T = tokens.shape
        device = tokens.device

        emb = self.embed(tokens)                           # [B, T, H]

        # Add image features to every step
        if features.dim() == 2:
            features = features.unsqueeze(1)               # [B,1,H]
        img_ctx = features.expand(B, T, features.size(-1)) # [B, T, H]
        x = emb + img_ctx                                  # [B, T, H]

        h0, c0 = self._normalize_hidden(hidden, B, device)
        out, (hn, cn) = self.lstm(self.dropout(x), (h0, c0))  # out: [B, T, H]
        logits = self.fc(self.dropout(out))                # [B, T, V]
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
        inputs = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)  # [B,1]
        hidden = None
        all_logits = []

        for _ in range(max_len):
            logits, hidden = self.forward(inputs[:, -1:], hidden, features=features)  # step
            step_logits = logits[:, -1:, :]           # [B,1,V]
            all_logits.append(step_logits)
            next_tok = step_logits.argmax(-1)         # [B,1]
            inputs = torch.cat([inputs, next_tok], dim=1)
            if (next_tok.squeeze(1) == eos_idx).all():
                break

        return inputs, torch.cat(all_logits, dim=1) if all_logits else torch.zeros(B, 0, self.vocab_size, device=device)


# -------------------------
# Main Net
# -------------------------
class Net(nn.Module):
    """
    Executable image captioning model with an encoder (self.cnn) and decoder (self.rnn).
    Preserves the original class/name and the forward signature used by your runner.
    """
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device

        # Robustly infer channels and vocab size
        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # Hyperparameters
        self.hidden_size = int(prm.get("hidden_size", 768))
        self.num_layers = int(prm.get("num_layers", 1))
        self.dropout = float(prm.get("dropout", 0.1))
        self.pad_idx = int(prm.get("pad_idx", 0))
        self.sos_idx = int(prm.get("sos_idx", 1))
        self.eos_idx = int(prm.get("eos_idx", 2))
        self.max_len = int(prm.get("max_len", 20))

        # Encoder and Decoder
        self.cnn = CNNEncoderTokens(self.in_channels, self.hidden_size)  # -> [B,1,H]
        self.rnn = DecoderRNN(self.vocab_size, self.hidden_size, num_layers=self.num_layers,
                              dropout=self.dropout, pad_idx=self.pad_idx)

        # Training helpers
        self.criteria: Tuple[nn.Module, ...] = ()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.to(self.device)

    # ----- API helpers -----
    @staticmethod
    def _infer_in_channels(in_shape) -> int:
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) == 4:
                return int(in_shape[1])  # (B,C,H,W)
            if len(in_shape) == 3:
                return int(in_shape[0])  # (C,H,W)
        # Fallback
        return 3

    @staticmethod
    def _first_int(x) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return Net._first_int(x[0])
        raise ValueError(f"Cannot infer integer from out_shape={x}")

    def init_hidden(self, batch_size, device):
        # For LSTM: (num_layers, B, H)
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    # ----- Public training API -----
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm["lr"], betas=(prm.get("momentum", 0.9), 0.999)
        )

    def learn(self, train_data: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
        """
        train_data yields (images, captions)
        images:   [B,C,H,W]
        captions: [B,T] with SOS at index 0, PAD=self.pad_idx
        """
        if not self.criteria or self.optimizer is None:
            return  # no-op if not configured

        self.train()
        for images, captions in train_data:
            images = images.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)

            if captions.dim() == 3:
                captions = captions[:, 0, :]  # teacher forcing convention in your logs

            inputs = captions[:, :-1]  # teacher-forcing inputs
            targets = captions[:, 1:]  # next-token targets

            features = self.cnn(images)                      # [B,1,H]
            logits, _ = self.rnn(inputs, None, features=features)  # [B,T-1,V]

            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # ----- Forward -----
    def forward(self, images, captions=None, hidden_state=None):
        """
        If captions is provided (teacher forcing):
           images:   [B,C,H,W]
           captions: [B,T] or [B,1,T]  -> logits [B,T,V]
        Else (inference):
           returns greedy logits [B,T_gen,V]
        """
        images = images.to(self.device)
        features = self.cnn(images)  # [B,1,H]

        # Teacher forcing
        if captions is not None:
            captions = captions.to(self.device)
            if captions.dim() == 3:
                captions = captions[:, 0, :]
            if captions.size(1) <= 1:
                # Nothing to predict
                return torch.zeros(captions.size(0), 0, self.vocab_size, device=self.device), hidden_state

            inputs = captions[:, :-1]  # feed all but last
            logits, hidden_state = self.rnn(inputs, hidden_state, features=features)
            return logits, hidden_state

        # Inference (greedy)
        tokens, gen_logits = self.rnn.greedy_decode(
            features, max_len=self.max_len, sos_idx=self.sos_idx, eos_idx=self.eos_idx, device=self.device
        )
        return gen_logits, (None, None)


# --------------- Minimal self-test ---------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 3, 224, 224
    V = 5000

    in_shape = (B, C, H, W)
    out_shape = (V,)
    prm = dict(lr=1e-4, momentum=0.9, hidden_size=512, num_layers=1, dropout=0.1,
               pad_idx=0, sos_idx=1, eos_idx=2, max_len=16)

    model = Net(in_shape, out_shape, prm, device).to(device)
    model.train_setup(prm)

    # Teacher-forcing path
    imgs = torch.randn(B, C, H, W, device=device)
    caps = torch.randint(0, V, (B, 12), device=device)
    caps[:, 0] = prm["sos_idx"]
    logits, _ = model(imgs, caps)         # [B, T-1, V]
    print("Teacher-forcing logits:", logits.shape)

    # Inference path
    gen_logits, _ = model(imgs, captions=None)
    print("Greedy logits:", gen_logits.shape)
