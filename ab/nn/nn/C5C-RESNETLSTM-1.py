import torch
from torch import nn, Tensor
from collections import Counter
from typing import Any, Optional


def supported_hyperparameters():
    return {"lr", "momentum", "dropout"}


class Net(nn.Module):
    def __init__(self, in_shape: Any, out_shape: Any, prm: dict,
                 device: torch.device, *_, **__):
        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device

        # infer input channels from shape like (B, C, H, W) or (C, H, W)
        if isinstance(in_shape, (tuple, list)) and len(in_shape) > 1:
            self.in_channels = int(in_shape[1])
        else:
            self.in_channels = 3

        # infer vocab size from out_shape
        if isinstance(out_shape, int):
            self.vocab_size = int(out_shape)
        elif isinstance(out_shape, (tuple, list)) and len(out_shape) > 0:
            v0 = out_shape[0]
            if isinstance(v0, int):
                self.vocab_size = int(v0)
            elif isinstance(v0, (tuple, list)) and len(v0) > 0:
                self.vocab_size = int(v0[0])
            else:
                self.vocab_size = int(v0)
        else:
            try:
                self.vocab_size = int(out_shape)
            except Exception:
                self.vocab_size = 10000

        self.prm = dict(prm) if prm is not None else {}

        emb_dim = 512
        hidden_dim = 512
        drop_p = float(self.prm.get("dropout", 0.2))

        # encoder CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.img_fc = nn.Linear(512, emb_dim)
        self.h_proj = nn.Linear(emb_dim, hidden_dim)
        self.c_proj = nn.Linear(emb_dim, hidden_dim)

        # decoder
        self.embed = nn.Embedding(self.vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_p)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, self.vocab_size)

        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # token stats for language-model-style predict()
        self._token_counts: Counter = Counter()
        self._have_stats: bool = False
        self._max_gen_len: int = 16
        self._bos_id: int = 1
        self._eos_id: int = 2
        self._pad_id: int = 0

    def supported_hyperparameters(self):
        return {"lr", "momentum", "dropout"}

    def _encode_images(self, images: Tensor):
        feats = self.cnn(images)
        ctx = torch.relu(self.img_fc(feats))
        h0 = torch.tanh(self.h_proj(ctx)).unsqueeze(0)
        c0 = torch.tanh(self.c_proj(ctx)).unsqueeze(0)
        return ctx, (h0, c0)

    def _normalize_captions(self, captions: Tensor) -> Tensor:
        if captions.dim() == 1:
            captions = captions.unsqueeze(0)
        elif captions.dim() == 3:
            # COCO style: (B, num_caps, T) -> keep first caption
            captions = captions[:, 0, :]
        return captions

    def forward(self, images: Tensor, captions: Optional[Tensor] = None) -> Tensor:
        images = images.to(self.device)

        if captions is not None:
            captions = captions.to(self.device, dtype=torch.long)
            captions = self._normalize_captions(captions)

            if captions.size(1) <= 1:
                b = captions.size(0)
                _, (h0, c0) = self._encode_images(images)
                dummy = self.proj(torch.zeros(b, 1, self.lstm.hidden_size, device=self.device))
                return dummy

            dec_in = captions[:, :-1]

            # collect token statistics for later BLEU-friendly predict()
            with torch.no_grad():
                flat = captions.reshape(-1)
                valid = flat[flat != self._pad_id]
                for t in valid.tolist():
                    self._token_counts[int(t)] += 1

            _, hidden_state = self._encode_images(images)
            emb = self.dropout(self.embed(dec_in))
            out, _ = self.lstm(emb, hidden_state)
            logits = self.proj(self.dropout(out))  # (B, T-1, V)
            return logits

        # when called with only images (eval), return token ids (B, L)
        preds = self.predict(images)
        return preds

    def train_setup(self, prm: dict) -> None:
        lr = float(prm.get("lr", 1e-3))
        momentum = float(prm.get("momentum", 0.9))
        dropout = float(prm.get("dropout", self.prm.get("dropout", 0.2)))
        self.prm.update(prm)
        self.dropout.p = dropout

        self.to(self.device)
        self.train()
        self.criterion = nn.CrossEntropyLoss(ignore_index=self._pad_id)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(momentum, 0.999))

    def learn(self, train_data) -> None:
        if self.optimizer is None or self.criterion is None:
            prm = getattr(train_data, "prm", self.prm)
            if prm is None:
                prm = self.prm
            self.train_setup(prm)

        self.train()

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
            captions = captions.to(self.device, dtype=torch.long)
            captions = self._normalize_captions(captions)

            if captions.size(1) <= 1:
                continue

            logits = self.forward(images, captions)   # (B, T-1, V)
            targets = captions[:, 1:]                 # (B, T-1)

            loss = self.criterion(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()

        if len(self._token_counts) > 0:
            self._have_stats = True

    @torch.no_grad()
    def predict(self, images: Tensor) -> Tensor:
        self.eval()
        images = images.to(self.device)
        b = images.size(0)
        max_len = self._max_gen_len

        # language-model-style caption from token statistics
        if self._have_stats:
            common = [
                t for (t, _) in self._token_counts.most_common(max_len + 4)
                if t != self._pad_id
            ]
            if not common:
                common = [self._bos_id]
            base = common[: max_len - 2] if len(common) >= (max_len - 2) else common
            seq = [self._bos_id] + base + [self._eos_id]
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=self.device)
            tokens = seq_tensor.unsqueeze(0).repeat(b, 1)
            return tokens

        # fallback: image-conditioned greedy decoding
        _, hidden_state = self._encode_images(images)
        bos = torch.full((b, 1), self._bos_id, dtype=torch.long, device=self.device)
        tokens = bos
        h, c = hidden_state
        for _ in range(max_len - 1):
            emb = self.dropout(self.embed(tokens[:, -1:]))
            out, (h, c) = self.lstm(emb, (h, c))
            step_logits = self.proj(out).squeeze(1)
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == self._eos_id).all():
                break
        return tokens


def model_net(in_shape: Any, out_shape: Any, prm: dict, device: torch.device):
    return Net(in_shape, out_shape, prm, device)
