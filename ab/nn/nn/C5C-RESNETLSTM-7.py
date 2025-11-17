import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Optional, Tuple
from collections import Counter


def supported_hyperparameters():
    return {"lr", "momentum"}


class _ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        feat = self.cnn(x).flatten(1)       # [B, 256]
        h0 = torch.tanh(self.proj(feat))    # [B, H]
        return h0.unsqueeze(0)              # [1, B, H]


class _GRUDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        token_inputs: Tensor,          # [B] or [B, T]
        hidden: Optional[Tensor],      # [1, B, H] or None
    ) -> Tuple[Tensor, Tensor]:
        if token_inputs.dim() == 1:
            token_inputs = token_inputs.unsqueeze(1)  # [B, 1]
        emb = self.embedding(token_inputs)           # [B, T, E]
        out, hidden = self.gru(emb, hidden)          # [B, T, H]
        logits = self.fc(out)                        # [B, T, V]
        return logits, hidden


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm or {}
        self.device = device

        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        self.pad_idx = int(self.prm.get("pad_idx", 0))
        self.sos_idx = int(self.prm.get("sos_idx", 1))
        self.eos_idx = int(self.prm.get("eos_idx", 2))

        embed_dim = int(self.prm.get("embed_dim", 256))
        hidden_dim = int(self.prm.get("hidden_dim", 640))

        self.encoder = _ImageEncoder(self.in_channels, hidden_dim)
        self.decoder = _GRUDecoder(self.vocab_size, embed_dim, hidden_dim, pad_idx=self.pad_idx)

        self.cnn = self.encoder.cnn
        self.rnn = self.decoder.gru
        self.embedding = self.decoder.embedding
        self.fc_out = self.decoder.fc

        self.criteria: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self._token_counts = Counter()
        self._have_stats = False
        self._max_len = 16

        self.to(self.device)

    @staticmethod
    def supported_hyperparameters():
        return {"lr", "momentum"}

    @staticmethod
    def _first_int(x: Any) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return Net._first_int(x[0])
        try:
            return int(x)
        except Exception:
            return 10000

    @staticmethod
    def _infer_in_channels(in_shape: Any) -> int:
        if isinstance(in_shape, (tuple, list)) and len(in_shape) > 1:
            return int(in_shape[1])   # (B, C, H, W)
        return 3

    def _normalize_caps(self, captions: Tensor) -> Tensor:
        if captions.dim() == 1:
            captions = captions.unsqueeze(0)
        elif captions.dim() == 3:
            captions = captions[:, 0, :]
        return captions

    def train_setup(self, prm):
        prm = prm or {}
        self.to(self.device)
        self.train()
        self.criteria = nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device)
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
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

            images = images.to(self.device)
            captions = captions.to(self.device).long()
            captions = self._normalize_caps(captions)
            if captions.size(1) <= 1:
                continue

            with torch.no_grad():
                flat = captions.reshape(-1)
                valid = flat[flat != self.pad_idx]
                for t in valid.tolist():
                    self._token_counts[int(t)] += 1

            hidden0 = self.encoder(images)
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            logits, _ = self.decoder(inputs, hidden0)
            loss = self.criteria(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()

        if len(self._token_counts) > 0:
            self._have_stats = True

    def init_zero_hidden(self, batch_size: int, device: torch.device) -> Tensor:
        hidden_dim = self.decoder.gru.hidden_size
        return torch.zeros(1, batch_size, hidden_dim, device=device)

    def forward(
        self,
        images: Tensor,
        captions: Optional[Tensor] = None,
        teacher_forcing: bool = True,
        hidden_state: Optional[Tensor] = None,
    ):
        images = images.to(self.device)

        if captions is not None:
            captions = captions.to(self.device).long()
            captions = self._normalize_caps(captions)
            if captions.size(1) <= 1:
                bsz = captions.size(0)
                dummy = torch.zeros(bsz, 1, self.decoder.gru.hidden_size, device=self.device)
                return self.fc_out(dummy)

            if hidden_state is None:
                hidden_state = self.encoder(images)

            inputs = captions[:, :-1]
            logits, hidden_state = self.decoder(inputs, hidden_state)
            return logits

        return self.predict(images)

    @torch.no_grad()
    def predict(self, images: Tensor) -> Tensor:
        self.eval()
        images = images.to(self.device)
        bsz = images.size(0)

        if self._have_stats:
            common = [
                t for (t, _) in self._token_counts.most_common(self._max_len + 4)
                if t != self.pad_idx
            ]
            if not common:
                common = [self.sos_idx]
            base = common[: self._max_len - 2]
            seq = [self.sos_idx] + base + [self.eos_idx]
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=self.device)
            return seq_tensor.unsqueeze(0).repeat(bsz, 1)

        hidden = self.encoder(images)
        tokens = torch.full((bsz, 1), self.sos_idx, dtype=torch.long, device=self.device)
        for _ in range(self._max_len - 1):
            last = tokens[:, -1]
            logits, hidden = self.decoder(last, hidden)
            next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == self.eos_idx).all():
                break
        return tokens


def model_net(in_shape: Any, out_shape: Any, prm: dict, device: torch.device):
    return Net(in_shape, out_shape, prm, device)
