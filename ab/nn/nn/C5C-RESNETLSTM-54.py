import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, input_channels: int = 3, output_channels: int = 640):
        super().__init__()
        self.layer1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.layer4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.layer5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Conv2d(1024, output_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.relu(self.layer6(x))
        # Pool to 1x1 so the encoder returns a [B, output_channels] vector
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x


class LSTMDecoderWithAttention(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, dropout: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Project token embeddings to hidden size before concatenation
        self.fc_embed = nn.Linear(embed_size, hidden_size)
        # Input to the cell is [projected_embed (hidden_size) ; memory (hidden_size)]
        self.lstm = nn.LSTMCell(hidden_size + hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        return h, c

    def forward(
        self,
        inputs: torch.Tensor,                # [B, T] or [B, 1]
        hidden_state=None,                   # (h, c) each [B, H]
        memory: torch.Tensor | None = None   # [B, H]
    ):
        """
        Returns:
            logits: [B, T, V]
            hidden: (h, c) each [B, H]
        """
        assert memory is not None, "memory (encoder features) must be provided"
        B = inputs.size(0)
        device = inputs.device
        h, c = hidden_state if hidden_state is not None else self._init_hidden(B, device)

        # Ensure 2D [B, T]
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(1)

        outputs = []
        for t in range(inputs.size(1)):
            tokens_t = inputs[:, t]  # [B]
            emb = self.embedding(tokens_t)           # [B, E]
            emb = self.fc_embed(emb)                 # [B, H]
            cat = torch.cat([emb, memory], dim=1)    # [B, 2H]
            h, c = self.lstm(cat, (h, c))            # each [B, H]
            logits_t = self.fc_out(self.dropout(h))  # [B, V]
            outputs.append(logits_t.unsqueeze(1))    # [B, 1, V]

        logits = torch.cat(outputs, dim=1)           # [B, T, V]
        return logits, (h, c)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device

        # Infer channels and vocab size robustly
        self.in_channels = int(in_shape[1]) if isinstance(in_shape, (tuple, list)) else 3
        self.vocab_size = int(out_shape[0]) if isinstance(out_shape, (tuple, list)) else int(out_shape)

        # Model dims
        self.hidden_size = int(prm.get("hidden_size", 640))
        self.embed_size = int(prm.get("embed_size", 300))

        # Encoder and decoder
        self.encoder = EncoderCNN(input_channels=self.in_channels, output_channels=self.hidden_size)
        self.rnn = LSTMDecoderWithAttention(
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            dropout=float(prm.get("dropout", 0.5)),
        )

        self.to(self.device)

    # Training utilities
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(prm["lr"]), betas=(float(prm.get("momentum", 0.9)), 0.999)
        )

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)

            # teacher forcing inputs/targets
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            memory = self.encoder(images)  # [B, H]
            logits, _ = self.rnn(inputs, None, memory)  # [B, T-1, V]

            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # Inference / forward
    def forward(self, images, captions=None, hidden_state=None):
        memory = self.encoder(images)  # [B, H]

        if captions is not None:
            inputs = captions[:, :-1]
            logits, hidden_state = self.rnn(inputs, hidden_state, memory)
            assert logits.shape[1] == inputs.shape[1]
            return logits, hidden_state
        else:
            # Greedy decoding
            B = images.size(0)
            sos_index = int(1)
            max_len = int(20)

            tokens = torch.full((B, 1), sos_index, dtype=torch.long, device=self.device)  # [B, 1]
            hidden_state = None
            logits_list = []

            for _ in range(max_len):
                step_logits, hidden_state = self.rnn(tokens[:, -1:], hidden_state, memory)  # [B, 1, V]
                logits_list.append(step_logits)
                next_token = step_logits.argmax(dim=-1)  # [B, 1]
                tokens = torch.cat([tokens, next_token], dim=1)

            logits = torch.cat(logits_list, dim=1)  # [B, max_len, V]
            return logits, hidden_state


def supported_hyperparameters():
    return {"lr", "momentum"}
