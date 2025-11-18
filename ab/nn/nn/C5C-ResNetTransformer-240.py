import torch
import torch.nn as nn
from typing import Optional


def supported_hyperparameters():
    return {"lr", "momentum"}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device

        if isinstance(out_shape, (tuple, list)):
            self.vocab_size = int(out_shape[0])
        else:
            self.vocab_size = int(out_shape)

        self.hidden_size = int(prm.get("hidden_size", 768))

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.gru = nn.GRU(
            input_size=self.hidden_size * 2,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

        self.criterion: Optional[nn.Module] = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def train_setup(self, prm):
        self.to(self.device)
        lr = float(prm["lr"])
        momentum = float(prm.get("momentum", 0.9))
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(momentum, 0.999),
        )
        self.criterion = self.criterion.to(self.device)

    def forward(self, inputs, hidden_state, features):
        inputs = inputs.to(self.device)
        if features is not None:
            features = features.to(self.device)

        emb = self.embedding(inputs)  # [B, T, H]

        if features is None:
            ctx = torch.zeros(
                emb.size(0),
                1,
                self.hidden_size,
                device=self.device,
                dtype=emb.dtype,
            )
        else:
            if features.dim() == 2:
                features = features.unsqueeze(1)
            ctx = features.mean(dim=1, keepdim=True)

        ctx_exp = ctx.expand(-1, emb.size(1), -1)
        dec_in = torch.cat([emb, ctx_exp], dim=-1)

        outputs, hidden_state = self.gru(dec_in, hidden_state)
        logits = self.fc_out(outputs)
        return logits, hidden_state

    def learn(self, inputs, hidden_state, features):
        assert self.optimizer is not None
        logits, hidden_state = self.forward(inputs, hidden_state, features)

        shift_logits = logits[:, :-1, :]        # [B, T-1, V]
        shift_targets = inputs[:, 1:]           # [B, T-1]

        loss = self.criterion(
            shift_logits.reshape(-1, self.vocab_size),
            shift_targets.reshape(-1),
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 3.0)
        self.optimizer.step()

        return loss
