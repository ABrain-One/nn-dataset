import torch
import torch.nn as nn
from typing import Tuple, Optional, Any


def supported_hyperparameters() -> set:
    return {"lr", "momentum"}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device
        self.prm = prm or {}

        # ---- basic shape info ----
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)
        self.hidden_size = int(self.prm.get("hidden_size", 768))

        # common attributes some parts of the framework expect
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # special tokens / decode length
        self.pad_idx = int(self.prm.get("pad_idx", 0))
        self.sos_idx = int(self.prm.get("sos", self.prm.get("sos_idx", 1)))
        self.eos_idx = int(self.prm.get("eos", self.pad_idx))
        self.max_len = int(self.prm.get("max_length", self.prm.get("max_len", 20)))

        # ---- encoder (CNN -> fixed-size feature -> hidden init) ----
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # make encoder output independent of input spatial size
        self.adapt_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.final_proj = nn.Linear(512 * 4 * 4, self.hidden_size)

        # ---- decoder (LSTM) ----
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_idx)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

        # training helpers (populated in train_setup)
        self.criteria = None
        self.criterion = None
        self.optimizer = None

        self.to(self.device)

    # ----------------- API helpers -----------------
    def train_setup(self, prm):
        prm = prm or {}
        self.to(self.device)
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device)
        self.criteria = (self.criterion,)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data: Any):
        self.train()
        if self.optimizer is None:
            self.train_setup(getattr(train_data, "prm", self.prm))

        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device).long()

            # handle [B,1,T] -> [B,T]
            if captions.dim() == 3 and captions.size(1) == 1:
                captions = captions[:, 0, :]

            if captions.dim() != 2 or captions.size(1) <= 1:
                continue

            inputs = captions[:, :-1]   # teacher forcing input
            targets = captions[:, 1:]   # predict next token

            self.optimizer.zero_grad(set_to_none=True)
            logits, _ = self.forward(images, inputs, None)
            loss = self.criterion(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # ----------------- encoder helpers -----------------
    def _encode_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device)
        feats = self.cnn(images)
        feats = self.adapt_pool(feats)           # [B, 512, 4, 4]
        feats = feats.flatten(1)                 # [B, 512*4*4]
        feats = self.final_proj(feats)           # [B, H]
        h0 = torch.tanh(feats).unsqueeze(0)      # [1, B, H]
        c0 = torch.zeros_like(h0)                # [1, B, H]
        return h0, c0

    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(1, batch, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch, self.hidden_size, device=device)
        return h0, c0

    # ----------------- Forward -----------------
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training:  images [B,C,H,W], captions [B,T_in] -> (logits [B,T_in,V], (h,c))
        Inference: images [B,C,H,W], captions=None -> (generated_ids [B,<=max_len], (h,c))
        """
        images = images.to(self.device)

        if hidden_state is None:
            hidden_state = self._encode_images(images)

        # teacher-forcing path
        if captions is not None:
            captions = captions.to(self.device).long()
            embedded = self.embedding(captions)                 # [B, T, H]
            output, (h_n, c_n) = self.rnn(embedded, hidden_state)
            logits = self.fc_out(output)                        # [B, T, V]
            return logits, h_n

        # greedy decode path (no captions provided)
        B = images.size(0)
        h, c = hidden_state
        tokens = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)
        generated = []

        for _ in range(self.max_len):
            emb = self.embedding(tokens[:, -1:])                # [B,1,H]
            out, (h, c) = self.rnn(emb, (h, c))                 # [B,1,H]
            step_logits = self.fc_out(out[:, -1, :])            # [B,V]
            next_tok = step_logits.argmax(dim=-1, keepdim=True) # [B,1]
            generated.append(next_tok)
            tokens = torch.cat([tokens, next_tok], dim=1)

            if (next_tok == self.eos_idx).all():
                break

        if generated:
            gen = torch.cat(generated, dim=1)                   # [B, L]
        else:
            gen = torch.empty(B, 0, dtype=torch.long, device=self.device)

        return gen, h

    # ----------------- Utility -----------------
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

    @staticmethod
    def _infer_in_channels(in_shape: Any) -> int:
        if isinstance(in_shape, (tuple, list)):
            # (C,H,W)
            if len(in_shape) == 3 and isinstance(in_shape[0], int):
                return int(in_shape[0])
            # (B,C,H,W)
            if len(in_shape) >= 2 and isinstance(in_shape[1], int):
                return int(in_shape[1])
        return 3
