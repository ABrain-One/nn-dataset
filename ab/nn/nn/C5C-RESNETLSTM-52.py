import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


def _first_int(x):
    """Extract first int from possibly nested tuple/list like (vocab,), [[vocab]], etc."""
    return _first_int(x[0]) if isinstance(x, (tuple, list)) else int(x)


class LSTMDecoder(nn.Module):
    """Small wrapper so we can access .embedding / call the module / and then a final .linear."""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, x_emb: torch.Tensor, hidden=None):
        # x_emb: (B, T, embedding_dim)
        return self.lstm(x_emb, hidden)


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape

        # Resolve shapes
        self.in_channels = int(in_shape[1]) if isinstance(in_shape, (tuple, list)) and len(in_shape) > 1 else 3
        self.vocab_size = _first_int(out_shape)

        # Model dims
        self.hidden_size = int(prm.get("hidden_size", max(256, min(768, self.vocab_size))))
        self.max_len = int(prm.get("max_len", 20))
        self.sos_idx = int(prm.get("sos_idx", 1))
        self.eos_idx = int(prm.get("eos_idx", self.vocab_size - 1))

        # -------- Encoder (simple CNN) --------
        enc = nn.Sequential()
        enc.add_module("conv1", nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1))
        enc.add_module("bn1", nn.BatchNorm2d(64))
        enc.add_module("relu1", nn.ReLU(inplace=True))
        enc.add_module("pool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        enc.add_module("conv2", nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        enc.add_module("bn2", nn.BatchNorm2d(128))
        enc.add_module("relu2", nn.ReLU(inplace=True))
        enc.add_module("pool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        enc.add_module("conv3", nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        enc.add_module("bn3", nn.BatchNorm2d(256))
        enc.add_module("relu3", nn.ReLU(inplace=True))
        enc.add_module("conv4", nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        enc.add_module("bn4", nn.BatchNorm2d(512))
        enc.add_module("relu4", nn.ReLU(inplace=True))

        enc.add_module("avgpool", nn.AdaptiveAvgPool2d((1, 1)))
        enc.add_module("flatten", nn.Flatten())
        # A simple projection head (not strictly used by the decoder here, but kept to match structure)
        enc.add_module("linear", nn.Linear(512, self.vocab_size))
        self.cnn = enc

        # -------- Decoder (embedding + LSTM + linear) --------
        self.rnn = LSTMDecoder(vocab_size=self.vocab_size, embedding_dim=self.hidden_size, hidden_size=self.hidden_size)

        # Training helpers (defaults; can be reset by train_setup)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get("lr", 1e-4)),
            betas=(float(prm.get("momentum", 0.9)), 0.999),
        )

        self.to(self.device)

    # -------------------- API: training setup --------------------
    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm["lr"]),
            betas=(float(prm.get("momentum", 0.9)), 0.999),
        )

    # -------------------- API: learning loop ---------------------
    def learn(self, train_data) -> None:
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)

            # Normalize caption shape to (B, T)
            if captions.dim() == 3:
                captions = captions.argmax(dim=-1) if captions.size(-1) > 1 else captions.squeeze(-1)

            logits, _ = self.forward(images, captions)

            # Teacher-forced next-token prediction
            targets = captions[:, 1:] if captions.size(1) > 1 else captions[:, :0]
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # -------------------- Forward -------------------------------
    def forward(self, images, captions=None, hidden_state=None):
        batch_size = images.size(0)
        _ = self.cnn(images)  # keep the encoder in the graph even if not used downstream

        if captions is None:
            # Greedy generation
            if hidden_state is None:
                hidden_state = self.init_zero_hidden(batch_size, self.device)

            tokens = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=self.device)
            outputs = []

            for _step in range(self.max_len):
                emb = self.rnn.embedding(tokens[:, -1:].contiguous())  # (B, 1, E)
                out, hidden_state = self.rnn(emb, hidden_state)         # out: (B, 1, H)
                step_logits = self.rnn.linear(out)                      # (B, 1, V)
                outputs.append(step_logits)
                next_ids = step_logits.argmax(dim=-1)                   # (B, 1)
                tokens = torch.cat([tokens, next_ids], dim=1)
                if (next_ids == self.eos_idx).all():
                    break

            logits = torch.cat(outputs, dim=1) if outputs else images.new_zeros((batch_size, 0, self.vocab_size))
            return logits, hidden_state

        # Training (teacher forcing): predict next tokens for each position
        if hidden_state is None:
            hidden_state = self.init_zero_hidden(batch_size, self.device)

        # inputs: all tokens except the last -> targets: all tokens except the first
        inputs = captions[:, :-1] if captions.size(1) > 1 else captions[:, :0]
        emb = self.rnn.embedding(inputs)              # (B, T-1, E)
        out, hidden_state = self.rnn(emb, hidden_state)  # (B, T-1, H)
        logits = self.rnn.linear(out)                 # (B, T-1, V)

        return logits, hidden_state

    def init_zero_hidden(self, batch_size, device):
        # For LSTM: (h0, c0) with num_layers=1
        h0 = torch.zeros(1, batch_size, self.rnn.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.rnn.hidden_size, device=device)
        return (h0, c0)
