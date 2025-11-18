import torch
import torch.nn as nn


def supported_hyperparameters():
    return {'lr', 'momentum'}


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 768):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(512, hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.body(images)
        x = self.proj(x.flatten(1))
        return x.unsqueeze(1)


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device

        if isinstance(in_shape, (tuple, list)):
            in_channels = int(in_shape[1] if len(in_shape) > 1 else in_shape[0])
        else:
            in_channels = int(in_shape)

        if isinstance(out_shape, (tuple, list)):
            self.vocab_size = int(out_shape[0])
        else:
            self.vocab_size = int(out_shape)

        self.hidden_dim = 768

        self.encoder = CNNEncoder(in_channels, self.hidden_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = None

    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = float(prm['lr'])
        momentum = float(prm.get('momentum', 0.9))
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(momentum, 0.999),
        )
        self.criterion = self.criterion.to(self.device)

    def _norm_caps(self, caps: torch.Tensor) -> torch.Tensor:
        if caps.ndim == 3:
            if caps.size(1) == 1:
                caps = caps[:, 0, :]
            else:
                caps = caps[:, :, 0]
        return caps.long()

    def learn(self, train_data):
        assert self.optimizer is not None
        self.train()
        total_loss = 0.0
        steps = 0

        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)

            assert images.dim() == 4
            assert captions.dim() == 2

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            memory = self.encoder(images)
            ctx = memory.mean(dim=1, keepdim=True)

            emb = self.embedding(inputs)
            ctx_exp = ctx.expand(-1, emb.size(1), -1)
            dec_in = torch.cat([emb, ctx_exp], dim=-1)

            outputs, _ = self.gru(dec_in)
            logits = self.fc_out(outputs)

            loss = self.criterion(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

        return total_loss / max(steps, 1)

    def forward(self, images, captions=None, hidden_state=None, max_length: int = 20):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        ctx = memory.mean(dim=1, keepdim=True)
        batch_size = images.size(0)

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)

            assert captions.dim() == 2
            assert captions.size(0) == batch_size

            inputs = captions[:, :-1]

            emb = self.embedding(inputs)
            ctx_exp = ctx.expand(-1, emb.size(1), -1)
            dec_in = torch.cat([emb, ctx_exp], dim=-1)

            outputs, hidden_state = self.gru(dec_in, hidden_state)
            logits = self.fc_out(outputs)
            return logits, hidden_state

        sos_idx = 1
        generated = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=self.device)

        for _ in range(max_length - 1):
            last_tokens = generated[:, -1:]
            emb = self.embedding(last_tokens)
            ctx_exp = ctx
            dec_in = torch.cat([emb, ctx_exp], dim=-1)
            outputs, hidden_state = self.gru(dec_in, hidden_state)
            step_logits = self.fc_out(outputs[:, -1, :])
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated, hidden_state
