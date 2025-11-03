import torch
import torch.nn as nn


def supported_hyperparameters():
    return {"lr", "momentum"}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # Infer vocabulary size robustly from nested tuples/lists
        vocab_size = out_shape
        while isinstance(vocab_size, (tuple, list)):
            vocab_size = vocab_size[0]
        self.vocab_size = int(vocab_size)

        # ----- Encoder (CNN backbone) -----
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.projector = nn.Linear(256, 768)
        self.relu_proj = nn.ReLU()

        # ----- Decoder (Embedding + LSTM + Linear head) -----
        self.embedding = nn.Embedding(self.vocab_size, 768)
        # Use nn.LSTM for sequence inputs (batch_first=True)
        self.lstm = nn.LSTM(input_size=768 + 768, hidden_size=768, num_layers=1, batch_first=True)
        self.fc = nn.Linear(768, self.vocab_size)
        self.dropout = nn.Dropout(float(prm.get("dropout", 0.3)))

        self.to(self.device)

    # ----- Training utilities -----
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(prm["lr"]), betas=(float(prm.get("momentum", 0.9)), 0.999)
        )

    def learn(self, train_data):
        self.train()
        for batch in train_data:
            # Support either dict batches or tuple batches
            if isinstance(batch, dict):
                images = batch["images"]
                captions = batch["captions"]
            else:
                images, captions = batch

            images = images.to(self.device)
            captions = captions.to(self.device)

            # Teacher forcing: input is all tokens except last; target is all tokens except first
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            logits, _ = self.forward(images, inputs)  # logits: [B, T-1, V]
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # ----- Forward / Inference -----
    def forward(self, images, captions=None, hidden_state=None):
        batch_size = images.size(0)
        images = images.to(self.device)

        # Encode image to a single 768-d feature per image
        x = self.features(images)
        x = self.global_pool(x)
        x = self.flatten(x)                   # [B, 256]
        img_features = self.relu_proj(self.projector(x))  # [B, 768]

        if captions is not None:
            # Training / teacher-forcing path
            captions = captions.to(self.device)  # [B, T]
            embedded = self.embedding(captions)  # [B, T, 768]
            img_rep = img_features.unsqueeze(1).expand(-1, embedded.size(1), -1)  # [B, T, 768]
            lstm_in = torch.cat([embedded, img_rep], dim=-1)  # [B, T, 1536]

            outputs, hidden_state = self.lstm(lstm_in, hidden_state)  # outputs: [B, T, 768]
            logits = self.fc(self.dropout(outputs))  # [B, T, V]
            return logits, hidden_state

        # Inference: greedy decoding
        sos_idx = 1
        max_len = 20
        tokens = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=self.device)  # [B, 1]
        logits_list = []
        # hidden_state is either None or a tuple (h0, c0) with shapes [1, B, 768]
        for _ in range(max_len):
            emb = self.embedding(tokens[:, -1:])        # [B, 1, 768]
            img_rep = img_features.unsqueeze(1)         # [B, 1, 768]
            lstm_in = torch.cat([emb, img_rep], dim=-1) # [B, 1, 1536]

            out, hidden_state = self.lstm(lstm_in, hidden_state)   # out: [B, 1, 768]
            step_logits = self.fc(self.dropout(out))               # [B, 1, V]
            logits_list.append(step_logits)

            next_token = step_logits.argmax(dim=-1)     # [B, 1]
            tokens = torch.cat([tokens, next_token], dim=1)

        logits = torch.cat(logits_list, dim=1)  # [B, max_len, V]
        return logits, hidden_state
