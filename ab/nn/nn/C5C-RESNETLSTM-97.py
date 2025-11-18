import torch
import torch.nn as nn
from typing import Any, Dict, Iterable, Optional, Tuple


def supported_hyperparameters():
    return {"lr", "momentum"}


class SimpleEncoderA97(nn.Module):
    """Lightweight CNN encoder -> single feature vector per image."""
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)      # [B,256,1,1]
        x = x.flatten(1)      # [B,256]
        return self.fc(x)     # [B,H]


class LSTMDecoderA97(nn.Module):
    """Embedding + LSTM + linear projection."""
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(1, batch, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch, self.hidden_size, device=device)
        return h0, c0

    def forward(
        self,
        inputs: torch.Tensor,   # [B,T]
        hidden: Tuple[torch.Tensor, torch.Tensor],
        features: Optional[torch.Tensor] = None,   # kept for API compatibility
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embed(inputs)             # [B,T,H]
        out, hidden = self.lstm(emb, hidden) # [B,T,H]
        logits = self.proj(out)              # [B,T,V]
        return logits, hidden


class Net(nn.Module):
    """
    CNN + LSTM captioning network.

    - Training: forward(images, captions) -> (logits [B,T,V], hidden_state)
    - Eval:     forward(images)           -> token IDs [B,S] (Tensor only, for BLEU)
    """
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.prm = prm or {}

        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # hyperparams
        self.hidden_size = int(self.prm.get("hidden_size", 640))
        self.sos_idx = int(self.prm.get("sos", 1))
        self.eos_idx = int(self.prm.get("eos", 0))
        self.max_len = int(self.prm.get("max_length", 20))
        self.grad_clip = float(self.prm.get("grad_clip", 3.0))

        # modules
        self.encoder = SimpleEncoderA97(self.in_channels, self.hidden_size)
        self.decoder = LSTMDecoderA97(self.hidden_size, self.vocab_size)
        self.h_init = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_init = nn.Linear(self.hidden_size, self.hidden_size)

        # For harness introspection
        self.cnn = self.encoder.body
        self.embedding = self.decoder.embed
        self.fc_out = self.decoder.proj

        # training helpers
        self.criterion: Optional[nn.Module] = None
        self.criteria = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    # -------- training plumbing --------
    def train_setup(self, prm: Dict[str, Any]):
        prm = prm or {}
        lr = float(prm.get("lr", self.prm.get("lr", 1e-3)))
        beta1 = float(prm.get("momentum", self.prm.get("momentum", 0.9)))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.criteria = (self.criterion,)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data: Iterable | Dict[str, torch.Tensor]):
        if self.optimizer is None or self.criterion is None:
            self.train_setup(self.prm)

        self.train()

        def _run_batch(images: torch.Tensor, captions: torch.Tensor):
            images = images.to(self.device).float()
            captions = captions.to(self.device).long()
            # normalize shape [B,1,T] -> [B,T]
            if captions.dim() == 3 and captions.size(1) == 1:
                captions = captions[:, 0, :]

            if captions.size(1) <= 1:
                return

            inputs = captions[:, :-1]   # [B,T-1]
            targets = captions[:, 1:]   # [B,T-1]

            logits, _ = self.forward(images, inputs)
            loss = self.criterion(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            self.optimizer.step()

        if isinstance(train_data, dict):
            _run_batch(train_data["images"], train_data["captions"])
        else:
            for batch in train_data:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    _run_batch(batch[0], batch[1])
                elif isinstance(batch, dict):
                    _run_batch(batch["images"], batch["captions"])

    # -------- forward / inference --------
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        images = images.to(self.device).float()
        feats = self.encoder(images)  # [B,H]

        if hidden_state is None:
            h0 = torch.tanh(self.h_init(feats)).unsqueeze(0)  # [1,B,H]
            c0 = torch.tanh(self.c_init(feats)).unsqueeze(0)  # [1,B,H]
            hidden_state = (h0, c0)

        # teacher forcing
        if captions is not None:
            captions = captions.to(self.device).long()
            if captions.dim() == 3 and captions.size(1) == 1:
                captions = captions[:, 0, :]
            logits, hidden_state = self.decoder(captions, hidden_state, feats)
            return logits, hidden_state

        # greedy decoding for BLEU
        B = images.size(0)
        seq = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)
        h = hidden_state

        for _ in range(self.max_len):
            step_logits, h = self.decoder(seq[:, -1:], h, feats)       # [B,1,V]
            next_tok = step_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=1)
            if (next_tok == self.eos_idx).all():
                break

        # IMPORTANT: return Tensor only (no tuple) so BLEU sees preds.dim()
        return seq

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        out = self.forward(images)
        if isinstance(out, tuple):
            out = out[0]
        return out

    # -------- helpers --------
    @staticmethod
    def _infer_in_channels(in_shape) -> int:
        # Handles (B,C,H,W) or (C,H,W), fallback 3
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) >= 4:
                return int(in_shape[1])
            if len(in_shape) == 3:
                return int(in_shape[0])
        return 3

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
