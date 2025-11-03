import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ResNetSpatialEncoder(nn.Module):
    """
    Lightweight CNN that returns spatial features for attention.
    Output: [B, L, H] where L = Hs * Ws spatial locations, H = hidden_size.
    """
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        in_ch = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3
        self.hidden_size = int(prm.get('hidden_size', 256)) if isinstance(prm, dict) else 256

        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),   nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),  nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            # Project channels to hidden_size; keep spatial dims for attention
            nn.Conv2d(256, self.hidden_size, kernel_size=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, C, H, W]
        returns features: [B, L, H] where H is hidden_size
        """
        x = self.body(images)                         # [B, H, Hs, Ws]
        x = x.flatten(2).transpose(1, 2).contiguous() # [B, L, H]
        return x


class LSTMDecoder(nn.Module):
    """
    Token decoder with dot-product attention over spatial features.
    inputs: [B, T] token ids
    features: [B, L, H] from encoder
    returns: logits [B, T, V]
    """
    def __init__(self, hidden_size: int, vocab_size: int, device: torch.device):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device

        self.embedder = nn.Embedding(vocab_size, hidden_size)
        # input to LSTMCell is [embed; context] => 2 * hidden_size
        self.lstm_cell = nn.LSTMCell(input_size=hidden_size * 2, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.attn_q = nn.Linear(hidden_size, hidden_size)

    def init_zero_hidden(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(batch_size, self.hidden_size, device=device)
        return (h0, c0)

    def _attend(self, h: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        h: [B, H], features: [B, L, H] -> context: [B, H]
        """
        q = self.attn_q(h)                                # [B, H]
        scores = (features * q.unsqueeze(1)).sum(dim=2)   # [B, L]
        weights = F.softmax(scores, dim=1)                # [B, L]
        context = (features * weights.unsqueeze(2)).sum(dim=1)  # [B, H]
        return context

    def forward(self, inputs: torch.Tensor, hidden_state, features: torch.Tensor):
        """
        inputs: [B, T]
        hidden_state: tuple(h, c) each [B, H]
        features: [B, L, H]
        """
        emb = self.embedder(inputs)  # [B, T, H]
        h, c = hidden_state
        outs = []

        for t in range(emb.size(1)):
            e_t = emb[:, t, :]                    # [B, H]
            ctx = self._attend(h, features)       # [B, H]
            x_t = torch.cat([e_t, ctx], dim=1)    # [B, 2H]
            h, c = self.lstm_cell(x_t, (h, c))    # [B, H]
            logits_t = self.fc(h)                 # [B, V]
            outs.append(logits_t)

        logits = torch.stack(outs, dim=1)         # [B, T, V]
        return logits, (h, c)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device

        # Heuristic: extract vocab size robustly from possible structures
        def _infer_vocab_size(os_):
            if isinstance(os_, (tuple, list)):
                first = os_[0]
                if isinstance(first, (tuple, list)):
                    return int(first[0])
                return int(first)
            return int(os_)

        self.vocab_size = _infer_vocab_size(out_shape)
        self.hidden_size = int(prm.get('hidden_size', 256)) if isinstance(prm, dict) else 256

        self.encoder = ResNetSpatialEncoder(in_shape, out_shape, prm, device)
        self.decoder = LSTMDecoder(self.hidden_size, self.vocab_size, device)

        # Training plumbing
        self.criteria = None
        self.optimizer = None

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(prm.get('lr', 1e-3)), betas=(float(prm.get('momentum', 0.9)), 0.999)
        )

    def learn(self, train_data):
        # Intentionally left minimal — your training loop likely lives elsewhere.
        self.train()
        if self.optimizer is None:
            raise RuntimeError("Call train_setup(prm) before learn().")
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)  # [B, T]

            feats = self.encoder(images)                         # [B, L, H]
            inputs = captions[:, :-1]                            # [B, T-1]
            targets = captions[:, 1:].contiguous()               # [B, T-1]

            logits, _ = self.decoder(inputs, self.decoder.init_zero_hidden(images.size(0), self.device), feats)  # [B, T-1, V]
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1), ignore_index=0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def forward(self, images, captions=None, hidden_state=None):
        """
        Training:  images [B,C,H,W], captions [B,T] -> logits [B,T-1,V]
        Inference: images [B,C,H,W] -> generated tokens [B,<=max_len]
        """
        images = images.to(self.device)
        feats = self.encoder(images)  # [B, L, H]

        if captions is not None:
            assert images.dim() == 4 and captions.dim() == 2, \
                "Input dimensions must be [B, C, H, W] and [B, T]"
            inputs = captions[:, :-1]
            if hidden_state is None:
                hidden_state = self.decoder.init_zero_hidden(images.size(0), self.device)
            logits, hidden_state = self.decoder(inputs, hidden_state, feats)
            return logits, hidden_state

        # Greedy decoding
        batch = images.size(0)
        max_len = 20
        sos_idx, eos_idx = 0, 1
        tokens = torch.full((batch, 1), sos_idx, dtype=torch.long, device=self.device)
        hidden = self.decoder.init_zero_hidden(batch, self.device)
        outputs = []

        for _ in range(max_len):
            logits, hidden = self.decoder(tokens[:, -1:].contiguous(), hidden, feats)  # feed last token
            next_tok = logits[:, -1, :].argmax(dim=-1)  # [B]
            outputs.append(next_tok.unsqueeze(1))
            tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)
            # stop if all reached EOS
            if (next_tok == eos_idx).all():
                break

        return torch.cat(outputs, dim=1), hidden
