import torch
import torch.nn as nn
from typing import Optional


class MLPEncoder(nn.Module):
    def __init__(self, in_shape, hidden_dim: int = 640):
        super().__init__()
        c, h, w = int(in_shape[0]), int(in_shape[1]), int(in_shape[2])
        in_features = c * h * w
        mid = max(512, hidden_dim)
        self.output_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b = images.size(0)
        x = images.view(b, -1)
        x = self.net(x)
        return x.unsqueeze(1)  # [B,1,H]


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device

        if isinstance(in_shape, (tuple, list)):
            c = int(in_shape[0])
            h = int(in_shape[1])
            w = int(in_shape[2])
            self.in_shape = (c, h, w)
        else:
            self.in_shape = (int(in_shape), 224, 224)

        if isinstance(out_shape, (tuple, list)):
            self.vocab_size = int(out_shape[0])
        else:
            self.vocab_size = int(out_shape)

        self.hidden_dim = int(prm.get("hidden_dim", 640))

        self.encoder = MLPEncoder(self.in_shape, hidden_dim=self.hidden_dim)
        self.memory_dim = self.hidden_dim

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.gru = nn.GRU(
            input_size=self.hidden_dim * 2,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm["lr"],
            betas=(prm.get("momentum", 0.9), 0.999),
        )

    def _norm_caps(self, caps: torch.Tensor) -> torch.Tensor:
        if caps.ndim == 3:
            if caps.size(1) == 1:
                caps = caps[:, 0, :]
            else:
                caps = caps[:, :, 0]
        return caps.long()

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        steps = 0
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)

            b = images.size(0)
            assert images.dim() == 4, "Images must be 4D"
            c, h, w = self.in_shape
            assert images.shape[1] == c, "Channel mismatch"
            assert images.shape[2] == h and images.shape[3] == w, "Spatial size mismatch"

            inp = captions[:, :-1]
            tgt = captions[:, 1:]

            memory = self.encoder(images)  # [B,1,H]
            assert memory.dim() == 3 and memory.size(0) == b

            context = memory.mean(dim=1, keepdim=True)  # [B,1,H]
            emb = self.embedding(inp)  # [B,T-1,H]
            ctx = context.expand(-1, emb.size(1), -1)
            x = torch.cat([emb, ctx], dim=-1)  # [B,T-1,2H]

            out, _ = self.gru(x)
            logits = self.fc_out(out)  # [B,T-1,V]

            loss = self.criteria[0](
                logits.reshape(-1, self.vocab_size),
                tgt.reshape(-1),
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

        return total_loss / max(steps, 1)

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
    ):
        images = images.to(self.device, dtype=torch.float32)
        b = images.size(0)
        memory = self.encoder(images)
        assert memory.shape == (b, 1, self.hidden_dim)
        context = memory.mean(dim=1, keepdim=True)

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)
            inp = captions[:, :-1]
            emb = self.embedding(inp)
            ctx = context.expand(-1, emb.size(1), -1)
            x = torch.cat([emb, ctx], dim=-1)
            out, hidden_state = self.gru(x, hidden_state)
            logits = self.fc_out(out)
            return logits, hidden_state

        batch_size = images.size(0)
        max_len = 20
        sos_idx = 1
        generated = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=self.device
        )
        hidden_state = None
        for _ in range(max_len - 1):
            emb = self.embedding(generated[:, -1:])
            ctx = context
            x = torch.cat([emb, ctx], dim=-1)
            out, hidden_state = self.gru(x, hidden_state)
            step_logits = self.fc_out(out[:, -1, :])
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated, hidden_state


def supported_hyperparameters():
    return {"lr", "momentum"}
