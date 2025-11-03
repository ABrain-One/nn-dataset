import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Iterable


def supported_hyperparameters():
    return {"lr", "momentum"}


# -------------------------
# ResNet-ish Encoder -> [B, 1, H]
# -------------------------
class Encoder(nn.Module):
    """CNN encoder that outputs one image token per image."""
    def __init__(self, in_channels: int, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        def block(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.layer1 = block(64, 128, stride=2)
        self.layer2 = block(128, 256, stride=2)
        self.layer3 = block(256, 512, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, hidden_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)          # [B, 512, H', W']
        x = self.pool(x).flatten(1)  # [B, 512]
        x = self.proj(x)             # [B, H]
        return x.unsqueeze(1)        # [B, 1, H]


# -------------------------
# LSTM Decoder -> logits
# -------------------------
class Decoder(nn.Module):
    """Caption decoder with teacher forcing + greedy inference."""
    def __init__(self, vocab_size: int, hidden_size: int = 768, num_layers: int = 1, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.pad_idx = int(pad_idx)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

    def _init_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=device)
        return h0, c0

    def forward(
        self,
        tokens: torch.Tensor,                              # [B, T]
        image_tokens: torch.Tensor,                        # [B, 1, H]
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T = tokens.shape
        device = tokens.device

        x = self.embedding(tokens)                         # [B, T, H]
        # add image embedding to every step as a simple context
        img_ctx = image_tokens.expand(B, T, image_tokens.size(-1))
        x = self.dropout(x + img_ctx)

        if hidden is None:
            hidden = self._init_hidden(B, device)

        out, hidden = self.lstm(x, hidden)                 # [B, T, H]
        logits = self.fc_out(self.dropout(out))            # [B, T, V]
        return logits, hidden

    @torch.no_grad()
    def greedy(
        self,
        image_tokens: torch.Tensor,                        # [B, 1, H]
        sos_idx: int,
        eos_idx: int,
        max_len: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = image_tokens.size(0)
        tokens = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
        hidden = None
        logits_all = []

        for _ in range(max_len):
            step_logits, hidden = self.forward(tokens[:, -1:], image_tokens, hidden)
            logits_all.append(step_logits)                 # [B,1,V]
            next_tok = step_logits.argmax(-1)              # [B,1]
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok.squeeze(1) == eos_idx).all():
                break

        logits = torch.cat(logits_all, dim=1) if logits_all else torch.zeros(B, 0, self.vocab_size, device=device)
        return tokens, logits


# -------------------------
# Main Net
# -------------------------
class Net(nn.Module):
    """
    Image Captioning Net (Encoder + Decoder).
    - Encoder: self.encoder
    - Decoder: self.decoder
    Exposed API:
      - train_setup(prm)
      - learn(train_iter)
      - forward(images, captions|None) -> (logits, hidden_state)
    """
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device

        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)

        # hyperparams
        self.hidden_size = int(prm.get("hidden_dim", prm.get("hidden_size", 768)))
        self.num_layers = int(prm.get("num_layers", 1))
        self.dropout = float(prm.get("dropout", 0.1))
        self.pad_idx = int(prm.get("pad_idx", 0))
        self.sos_idx = int(prm.get("sos_idx", 1))
        self.eos_idx = int(prm.get("eos_idx", 2))
        self.max_len = int(prm.get("max_length", 20))

        # modules
        self.encoder = Encoder(self.in_channels, self.hidden_size)
        self.decoder = Decoder(self.vocab_size, self.hidden_size, self.num_layers, self.dropout, self.pad_idx)

        self.criteria: Tuple[nn.Module, ...] = ()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.to(self.device)

    @staticmethod
    def _infer_in_channels(in_shape) -> int:
        # supports (B,C,H,W) or (C,H,W)
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) >= 2:
                return int(in_shape[1] if len(in_shape) == 4 else in_shape[0])
        return 3

    @staticmethod
    def _first_int(x) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return Net._first_int(x[0])
        raise ValueError(f"Could not parse vocab size from out_shape={x}")

    # ---- training helpers ----
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm["lr"], betas=(prm.get("momentum", 0.9), 0.999)
        )

    def learn(self, train_iter: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
        """train_iter yields (images, captions) with captions shape [B,T] or [B,1,T]"""
        if not self.criteria or self.optimizer is None:
            return
        self.train()

        for images, captions in train_iter:
            images = images.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)
            if captions.dim() == 3:
                captions = captions[:, 0, :]

            if captions.size(1) <= 1:
                continue

            inputs = captions[:, :-1]       # teacher forcing input
            targets = captions[:, 1:]       # next-token targets

            img_tok = self.encoder(images)                       # [B,1,H]
            logits, _ = self.decoder(inputs, img_tok, None)      # [B,T-1,V]

            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # ---- forward ----
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        """
        With captions (teacher forcing):
            returns (logits [B,T-1,V], hidden_state)
        Without captions (inference, greedy):
            returns (logits [B,T_gen,V], (None,None))
        """
        images = images.to(self.device)
        img_tok = self.encoder(images)  # [B,1,H]

        if captions is not None:
            captions = captions.to(self.device)
            if captions.dim() == 3:
                captions = captions[:, 0, :]
            if captions.size(1) <= 1:
                empty = torch.zeros(captions.size(0), 0, self.vocab_size, device=self.device)
                return empty, hidden_state
            inputs = captions[:, :-1]
            logits, hidden_state = self.decoder(inputs, img_tok, hidden_state)
            return logits, hidden_state

        # Greedy inference
        _, gen_logits = self.decoder.greedy(
            image_tokens=img_tok,
            sos_idx=self.sos_idx,
            eos_idx=self.eos_idx,
            max_len=self.max_len,
            device=self.device,
        )
        return gen_logits, (None, None)


# ---- quick self-test (safe to keep or delete) ----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 3, 224, 224
    V = 5000
    in_shape = (B, C, H, W)
    out_shape = (V,)
    prm = dict(lr=1e-4, momentum=0.9, hidden_dim=768, num_layers=1, dropout=0.1,
               pad_idx=0, sos_idx=1, eos_idx=2, max_length=16)

    net = Net(in_shape, out_shape, prm, device).to(device)
    net.train_setup(prm)

    imgs = torch.randn(B, C, H, W, device=device)
    caps = torch.randint(0, V, (B, 12), device=device)
    caps[:, 0] = prm["sos_idx"]

    # Teacher forcing path
    logits, _ = net(imgs, caps)
    print("Teacher-forcing logits:", tuple(logits.shape))  # (B, T-1, V)

    # Inference path
    gen_logits, _ = net(imgs, captions=None)
    print("Greedy logits:", tuple(gen_logits.shape))       # (B, <=max_length, V)
