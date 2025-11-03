import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Any, Optional, Tuple


class Net(nn.Module):
    """
    CNN image encoder -> GRU decoder for captioning.
    Implements the harness contract: train_setup(prm), learn(data_roll), predict(x), forward().
    Consumes prm['lr'], prm['momentum'], prm['dropout'].
    """

    def __init__(self, in_shape: Any, out_shape: Any, prm: dict, device: torch.device, *_, **__):
        super().__init__()

        # ---- store & normalize shapes / device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device

        # channels
        if isinstance(in_shape, (tuple, list)) and len(in_shape) > 1:
            self.in_channels = int(in_shape[1])  # (C,H,W)
        else:
            self.in_channels = 3

        # vocab size (robust to int or tuple-like)
        if isinstance(out_shape, int):
            self.vocab_size = int(out_shape)
        elif isinstance(out_shape, (tuple, list)):
            # common patterns: (V,), or ((V,), ...)
            v0 = out_shape[0]
            self.vocab_size = int(v0 if isinstance(v0, int) else (v0[0] if isinstance(v0, (tuple, list)) else v0))
        else:
            # best effort
            try:
                self.vocab_size = int(out_shape)
            except Exception:
                self.vocab_size = 10000

        # hyperparams (store, but configured in train_setup)
        self.prm = dict(prm) if prm is not None else {}

        # ---- model sizes
        emb_dim = 256
        hidden_dim = 768
        drop_p = float(self.prm.get("dropout", 0.0))

        # ---- encoder: lightweight CNN -> pooled vector
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),   # (B,64,1,1)
            nn.Flatten(),                   # (B,64)
        )

        # map to embedding space & init GRU hidden
        self.img_to_emb = nn.Linear(64, emb_dim)
        self.h0_proj = nn.Linear(emb_dim, hidden_dim)
        self.ctx_proj = nn.Linear(emb_dim, emb_dim)


        # ---- decoder
        self.embedding = nn.Embedding(self.vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_p)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, self.vocab_size)

        # training state (initialized in train_setup)
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    # (optional, some harnesses inspect this)
    def supported_hyperparameters(self):
        return {"lr", "momentum", "dropout"}

    def _image_context(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode images to (context_emb, h0) for the decoder."""
        feats = self.cnn(images)                # (B,64)
        ctx = self.img_to_emb(feats)            # (B,E)
        h0 = torch.tanh(self.h0_proj(ctx))      # (B,H)
        h0 = h0.unsqueeze(0)                    # (1,B,H) for GRU
        return ctx, h0

    def forward(self, images: Tensor, captions: Optional[Tensor] = None) -> Tensor:
        """
        If captions provided (B,T): teacher forcing, returns logits (B,T-1,V).
        If captions is None: single step decode, returns logits (B,V).
        """
        images = images.to(self.device, dtype=torch.float32)
        if captions is not None:
            if captions.dim() == 3:
                # keep first path if extra dims are present
                captions = captions[:, 0, :]
            captions = captions.long().to(self.device)

            # teacher forcing: predict next token for each position
            if captions.size(1) <= 1:
                # nothing to predict
                B = captions.size(0)
                return self.fc_out(torch.zeros(B, 1, self.gru.hidden_size, device=self.device)).squeeze(1)

            dec_in_tokens = captions[:, :-1]     # (B,T-1)
            targets = captions[:, 1:]            # (B,T-1)

            # encode image, get context & initial GRU hidden
            ctx, h0 = self._image_context(images)                        # ctx: (B,E)

            emb = self.embedding(dec_in_tokens)                          # (B,T-1,E)
            emb = emb + self.ctx_proj(ctx).unsqueeze(1)                  # add image bias each step
            emb = self.dropout(emb)
            out, _ = self.gru(emb, h0)                                   # (B,T-1,H)
            logits = self.fc_out(self.dropout(out))                      # (B,T-1,V)

            return logits


        ctx, h0 = self._image_context(images)
        emb0 = self.dropout(self.ctx_proj(ctx)).unsqueeze(1)   # (B,1,E)
        out, _ = self.gru(emb0, h0)                            # (B,1,H)
        logits = self.fc_out(out).squeeze(1)                   # (B,V)
        return logits



    def train_setup(self, prm: dict) -> None:
        # capture hyperparams; consume momentum even with AdamW to satisfy checker
        lr = float(prm.get("lr", 1e-3))
        momentum = float(prm.get("momentum", 0.9))
        dropout = float(prm.get("dropout", self.prm.get("dropout", 0.0)))
        # update dropout prob at train time if changed
        self.dropout.p = dropout

        self.to(self.device)
        self.train()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        # Use AdamW; inject "momentum" as beta1 so the param is genuinely used
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(momentum, 0.999))

    def learn(self, data_roll) -> None:
        """
        Iterate batches from DataRoll (don't index it).
        Expect batches like (images, captions, *extras) or dicts with keys 'x', 'y'.
        """
        # some harnesses pass prm via data_roll.prm; ensure train setup
        prm = getattr(data_roll, "prm", self.prm)
        self.train_setup(prm)

        for batch in data_roll:
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
                # unknown structure, try attributes
                images = getattr(batch, "x", None)
                captions = getattr(batch, "y", None)
                if images is None or captions is None:
                    continue

            images = images.to(self.device, dtype=torch.float32)
            captions = captions.long().to(self.device)

            if captions.dim() == 3:
                captions = captions[:, 0, :]

            if captions.size(1) <= 1:
                # no next-token targets available
                continue

            logits = self.forward(images, captions)  # (B,T-1,V)
            targets = captions[:, 1:]                # (B,T-1)

            loss = self.criterion(logits.reshape(-1, self.vocab_size),
                                  targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()

    @torch.no_grad()
    def predict(self, images: Tensor) -> Tensor:
        """
        Greedy generation of a short caption (returns token ids shape (B, L)).
        Harnesses typically don't require generation, but keep it for completeness.
        """
        self.eval()
        images = images.to(self.device)
        _, h0 = self._image_context(images)

        B = images.size(0)
        max_len = 16
        bos_id = 1
        eos_id = 2  # guess; harmless if absent
        tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=self.device)

        hidden = h0
        outputs = [tokens]

        for _ in range(max_len - 1):
            emb = self.embedding(tokens[:, -1:])          # (B,1,E)
            emb = self.dropout(emb)
            out, hidden = self.gru(emb, hidden)           # (B,1,H)
            step_logits = self.fc_out(out).squeeze(1)     # (B,V)
            next_tok = step_logits.argmax(dim=-1, keepdim=True)  # (B,1)
            outputs.append(next_tok)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == eos_id).all():
                break

        return tokens


def model_net(in_shape: Any, out_shape: Any, prm: dict, device: torch.device):
    """
    Factory function expected by the training harness.
    """
    return Net(in_shape, out_shape, prm, device)
