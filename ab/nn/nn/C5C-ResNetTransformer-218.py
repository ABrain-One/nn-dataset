import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any, Iterable


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 768):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(512, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.proj(x.flatten(1))
        return x.unsqueeze(1)  # [B,1,H]


class DecoderGRU(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = hidden_dim

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        # GRUCell that takes [embed + context]
        self.gru = nn.GRUCell(self.embedding_dim + self.hidden_dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(
        self,
        inputs: torch.Tensor,      # [B,T] token indices
        hidden_state: torch.Tensor,  # [B,H]
        memory: torch.Tensor,      # [B,1,H]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = inputs.size()
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size, device=inputs.device)

        # single image vector per batch
        img_vec = memory.squeeze(1)  # [B,H]
        h_t = hidden_state

        for t in range(seq_len):
            x_t = inputs[:, t]
            embedded = self.embedding(x_t)  # [B,H]

            # Simple "attention": just concatenate embedded with img_vec
            combined = torch.cat([embedded, img_vec], dim=-1)  # [B,2H]
            h_t = self.gru(combined, h_t)                      # [B,H]
            logits_t = self.fc_out(h_t)                        # [B,V]
            outputs[:, t] = logits_t

        return outputs, h_t


class Net(nn.Module):
    def __init__(self, in_shape: Any, out_shape: Any, prm: dict, device: torch.device, *_, **__):
        super().__init__()
        self.device = device

        if isinstance(in_shape, (tuple, list)):
            in_channels = int(in_shape[0])
        else:
            in_channels = int(in_shape)

        if isinstance(out_shape, (tuple, list)):
            self.vocab_size = int(out_shape[0])
        else:
            self.vocab_size = int(out_shape)

        self.hidden_dim = prm.get("hidden_dim", 768)

        self.encoder = Encoder(in_channels=in_channels, hidden_dim=self.hidden_dim)
        self.decoder = DecoderGRU(vocab_size=self.vocab_size, hidden_dim=self.hidden_dim)

        self.criterion = None
        self.optimizer = None

    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = prm["lr"]
        momentum = prm.get("momentum", 0.9)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            betas=(momentum, 0.999),
        )

    def _norm_caps(self, caps: torch.Tensor) -> torch.Tensor:
        if caps.ndim == 3:
            if caps.size(1) == 1:
                caps = caps[:, 0, :]
            else:
                caps = caps[:, :, 0]
        return caps.long()

    def learn(self, train_data: Iterable):
        """
        train_data: iterable of (images, captions)
        """
        assert self.optimizer is not None, "Call train_setup(prm) before learn()"
        self.train()
        total_loss = 0.0
        steps = 0

        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)  # [B,T]

            inp = captions[:, :-1]   # [B,T-1]
            tgt = captions[:, 1:]    # [B,T-1]

            memory = self.encoder(images)        # [B,1,H]
            batch_size = images.size(0)
            h0 = torch.zeros(batch_size, self.hidden_dim, device=self.device)

            logits, h_t = self.decoder(inp, h0, memory)   # [B,T-1,V]

            loss = self.criterion(
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
        max_length: int = 20,
    ):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)  # [B,1,H]
        batch_size = images.size(0)

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)
            inp = captions[:, :-1]

            if hidden_state is None:
                hidden_state = torch.zeros(batch_size, self.hidden_dim, device=self.device)

            logits, hidden_state = self.decoder(inp, hidden_state, memory)
            return logits, hidden_state

        # Inference: greedy
        sos_idx = 1
        generated = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=self.device
        )
        hidden_state = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        for _ in range(max_length - 1):
            logits, hidden_state = self.decoder(generated[:, -1:].contiguous(), hidden_state, memory)
            step_logits = logits[:, -1, :]  # [B,V]
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated, hidden_state


def supported_hyperparameters():
    return {"lr", "momentum"}
