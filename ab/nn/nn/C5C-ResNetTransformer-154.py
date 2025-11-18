import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {"lr", "momentum"}


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        c = [64, 128, 256, hidden_size]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, c[0], 7, 2, 3, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(c[0], c[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[1], c[2], 3, 2, 1, bias=False),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[2], c[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(c[3]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1).contiguous()
        if x.size(1) > 196:
            x = x[:, :196, :]
        return x


class LSTMAttnDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_idx):
        super().__init__()
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(hidden_size + hidden_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.attn_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, memory, captions, hidden_state=None):
        emb = self.embedding(captions)
        b, t, _ = emb.shape
        if hidden_state is None:
            h0 = torch.zeros(1, b, self.hidden_size, device=emb.device)
            c0 = torch.zeros(1, b, self.hidden_size, device=emb.device)
            hidden_state = (h0, c0)
        h, c = hidden_state
        outputs = []
        for step in range(t):
            step_emb = emb[:, step:step+1, :]
            q = self.attn_proj(h[-1])[:, None, :]
            attn_scores = torch.bmm(q, memory.transpose(1, 2))
            attn_w = F.softmax(attn_scores, dim=-1)
            context = torch.bmm(attn_w, memory)
            lstm_in = torch.cat([step_emb, context], dim=-1)
            out, (h, c) = self.lstm(lstm_in, (h, c))
            logits = self.fc_out(out)
            outputs.append(logits)
        logits = torch.cat(outputs, dim=1)
        return logits, (h, c)

    def init_zero_hidden(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0, c0


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device
        self.prm = prm or {}

        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)
        self.hidden_size = int(self.prm.get("hidden_size", 768))
        self.pad_idx = int(self.prm.get("pad_idx", 0))
        self.sos_idx = int(self.prm.get("sos_idx", 1))
        self.eos_idx = int(self.prm.get("eos_idx", 2))
        self.max_len = int(self.prm.get("max_len", 16))

        self.encoder = ResNetEncoder(self.in_channels, self.hidden_size)
        self.decoder = LSTMAttnDecoder(self.vocab_size, self.hidden_size, self.pad_idx)

        self.cnn = self.encoder.cnn
        self.embedding = self.decoder.embedding
        self.fc_out = self.decoder.fc_out

        self.criterion = None
        self.optimizer = None

        self.to(self.device)

    @staticmethod
    def supported_hyperparameters():
        return {"lr", "momentum"}

    @staticmethod
    def _infer_in_channels(in_shape):
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) == 3 and isinstance(in_shape[0], int):
                return int(in_shape[0])
            if len(in_shape) >= 2 and isinstance(in_shape[1], int):
                return int(in_shape[1])
        return 3

    @staticmethod
    def _first_int(x):
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and x:
            return Net._first_int(x[0])
        return int(x)

    def _normalize_captions(self, captions):
        if captions.dim() == 1:
            captions = captions.unsqueeze(0)
        elif captions.dim() == 3:
            captions = captions[:, 0, :]
        return captions

    def train_setup(self, prm):
        prm = prm or {}
        self.to(self.device)
        self.train()
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device)
        self.criteria = (self.criterion,)
        lr = float(prm.get("lr", self.prm.get("lr", 1e-3)))
        beta1 = float(prm.get("momentum", self.prm.get("momentum", 0.9)))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        if not hasattr(self, "optimizer") or self.optimizer is None:
            self.train_setup(getattr(train_data, "prm", self.prm))

        self.train()
        for batch in train_data:
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
                images = getattr(batch, "x", None)
                captions = getattr(batch, "y", None)
                if images is None or captions is None:
                    continue

            images = images.to(self.device)
            captions = self._normalize_captions(captions.to(self.device).long())
            if captions.size(1) <= 1:
                continue

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            memory = self.encoder(images)
            logits, _ = self.decoder(memory, inputs)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images, captions=None, hidden_state=None):
        memory = self.encoder(images.to(self.device))
        if captions is not None:
            captions = self._normalize_captions(captions.to(self.device).long())
            logits, hidden_state = self.decoder(memory, captions, hidden_state)
            return logits, hidden_state

        b = images.size(0)
        h, c = self.decoder.init_zero_hidden(b, self.device)
        tokens = torch.full((b, 1), self.sos_idx, dtype=torch.long, device=self.device)
        for _ in range(self.max_len - 1):
            logits, (h, c) = self.decoder(memory, tokens[:, -1:].contiguous(), (h, c))
            step_logits = logits[:, -1, :]
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == self.eos_idx).all():
                break
        return tokens

    @torch.no_grad()
    def predict(self, images):
        return self.forward(images)


def model_net(in_shape, out_shape, prm, device):
    return Net(in_shape, out_shape, prm, device)
