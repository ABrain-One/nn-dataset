import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Any, Iterable


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.net(images)            # [B,H,1,1]
        x = x.view(x.size(0), 1, -1)    # [B,1,H]
        return x


class Net(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        out_shape: Any,
        prm: dict,
        device: torch.device,
        *_,
        **__,
    ) -> None:
        super().__init__()
        self.device = device

        # in_shape can be (C,H,W) or (B,C,H,W)-like; we only need channels.
        in_channels = int(in_shape[0])

        # out_shape can be int or tuple, use first element as vocab size.
        if isinstance(out_shape, (tuple, list)):
            self.vocab_size = int(out_shape[0])
        else:
            self.vocab_size = int(out_shape)

        self.hidden_dim = 768

        # Encoder
        self.encoder = CNNEncoder(in_channels=in_channels, hidden_dim=self.hidden_dim)

        # Decoder: GRU conditioned on image embedding as context
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.gru = nn.GRU(
            input_size=self.hidden_dim * 2,  # token + image context
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)

        self.criterion = None
        self.optimizer = None

    def train_setup(self, prm: dict) -> None:
        """Set up optimizer etc. prm must contain 'lr' and 'momentum'."""
        self.to(self.device)
        lr = prm["lr"]
        momentum = prm.get("momentum", 0.9)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def _norm_caps(self, captions: torch.Tensor) -> torch.Tensor:
        # Accept [B,T], [B,1,T], or [B,T,1] and return [B,T]
        if captions.ndim == 3:
            if captions.size(1) == 1:
                captions = captions[:, 0, :]
            else:
                captions = captions[:, :, 0]
        return captions.long()

    def _step_batch(self, images: torch.Tensor, captions: torch.Tensor):
        images = images.to(self.device, dtype=torch.float32)
        captions = captions.to(self.device)
        captions = self._norm_caps(captions)  # [B,T]

        # Teacher forcing: predict t+1 from 0..t
        inp = captions[:, :-1]  # [B,T-1]
        tgt = captions[:, 1:]   # [B,T-1]

        memory = self.encoder(images)  # [B,1,H]
        context = memory.mean(dim=1, keepdim=True)  # [B,1,H]

        emb = self.embedding(inp)  # [B,T-1,H]
        ctx = context.expand(-1, emb.size(1), -1)   # [B,T-1,H]

        dec_inp = torch.cat([emb, ctx], dim=-1)     # [B,T-1,2H]
        outputs, hidden_state = self.gru(dec_inp)   # [B,T-1,H]

        logits = self.fc_out(outputs)               # [B,T-1,V]
        loss = self.criterion(
            logits.reshape(-1, self.vocab_size),
            tgt.reshape(-1),
        )
        return logits, hidden_state, loss

    def learn(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Single-batch training helper (kept for compatibility).
        Returns (logits, hidden_state). Assumes optimizer already set via train_setup.
        """
        assert self.optimizer is not None, "Call train_setup(prm) before learn()"
        self.train()

        logits, hidden_state, loss = self._step_batch(images, captions)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 3.0)
        self.optimizer.step()
        return logits.detach(), hidden_state

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state=None,
        max_length: int = 20,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward for training & inference.
        Training: provide images & captions -> returns logits [B,T-1,V].
        Inference: provide images only -> returns sampled token ids [B,T].
        """
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)  # [B,1,H]
        context = memory.mean(dim=1, keepdim=True)  # [B,1,H]

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)
            inp = captions[:, :-1]  # [B,T-1]

            emb = self.embedding(inp)
            ctx = context.expand(-1, emb.size(1), -1)
            dec_inp = torch.cat([emb, ctx], dim=-1)

            outputs, hidden_state = self.gru(dec_inp, hidden_state)
            logits = self.fc_out(outputs)
            return logits, hidden_state

        # Inference: greedy generation
        batch_size = images.size(0)
        sos_idx = 1  # assume 1 = <SOS>
        generated = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=self.device
        )
        hidden_state = None

        for _ in range(max_length - 1):
            emb = self.embedding(generated[:, -1:])
            ctx = context
            dec_inp = torch.cat([emb, ctx], dim=-1)  # [B,1,2H]
            outputs, hidden_state = self.gru(dec_inp, hidden_state)
            step_logits = self.fc_out(outputs[:, -1, :])  # [B,V]
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated, hidden_state


def supported_hyperparameters():
    return {"lr", "momentum"}
