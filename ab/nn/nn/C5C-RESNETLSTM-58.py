import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


# ---------- Building blocks ----------
class SELayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.avg(x))
        return x * w


class PositionalEncoding(nn.Module):
    """Sinusoidal PE for batch_first=(B, T, D)."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # register as (1, T, D) so it broadcasts over batch
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return self.dropout(x + self.pe[:, :T, :])


# ---------- Encoder ----------
class SpatialAttentionEncoder(nn.Module):
    """Compact CNN encoder that returns a single image embedding (B, hidden_dim)."""
    def __init__(self, in_channels: int, hidden_dim: int = 768):
        super().__init__()
        c1, c2, c3, c4 = 64, 128, 256, 512
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            SELayer(c2),

            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            SELayer(c3),

            nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            SELayer(c4),

            nn.Conv2d(c4, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> (B, D, 1, 1)
        self.flatten = nn.Flatten()          # -> (B, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x  # (B, D)


# ---------- Decoder ----------
class TransformerCaptioningDecoder(nn.Module):
    """Transformer decoder (batch_first) that attends to a single image token."""
    def __init__(self, vocab_size: int, hidden_dim: int = 768, num_layers: int = 2,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.posenc = PositionalEncoding(hidden_dim, max_len=512, dropout=dropout)
        self.dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(self.dec_layer, num_layers=num_layers)
        self.readout = nn.Linear(hidden_dim, vocab_size)

    @staticmethod
    def subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        # (T, T) True for positions that should be masked
        mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, image_vec: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        image_vec: (B, D)   - encoded image embedding
        tokens:    (B, T)   - input token ids (teacher forcing)
        returns:   (B, T, V)
        """
        B, T = tokens.size(0), tokens.size(1)
        tgt = self.embed(tokens)                # (B, T, D)
        tgt = self.posenc(tgt)                  # (B, T, D)

        memory = image_vec.unsqueeze(1)         # (B, 1, D)
        tgt_mask = self.subsequent_mask(T, tokens.device)  # (T, T)
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)  # (B, T, D)
        return self.readout(out)                # (B, T, V)


# ---------- Full model ----------
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device
        prm = prm or {}

        # Infer channels and vocab size robustly
        def _infer_channels(shape):
            if isinstance(shape, (tuple, list)) and len(shape) >= 2:
                return int(shape[1])
            return int(shape if isinstance(shape, int) else 3)

        def _infer_vocab_size(shape):
            x = shape
            while isinstance(x, (tuple, list)):
                if len(x) == 0:
                    raise ValueError("out_shape is empty; cannot infer vocab size.")
                x = x[0]
            return int(x)

        self.in_shape = tuple(in_shape) if isinstance(in_shape, (tuple, list)) else (in_shape,)
        self.vocab_size = _infer_vocab_size(out_shape)
        in_channels = _infer_channels(in_shape)
        hidden_dim = int(prm.get("hidden_dim", 768))
        num_layers = int(prm.get("decoder_layers", 2))
        num_heads = int(prm.get("decoder_heads", 8))

        self.encoder = SpatialAttentionEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
        self.decoder = TransformerCaptioningDecoder(
            vocab_size=self.vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=float(prm.get("dropout", 0.1)),
        )

        # tokens
        self.sos_idx = int(prm.get("sos_idx", 1))
        self.eos_idx = int(prm.get("eos_idx", 2))
        self.max_len = int(prm.get("max_len", 20))

        self.to(self.device)

    # Training utilities
    def train_setup(self, prm):
        self.to(self.device)
        lr = float(prm.get("lr", 1e-3))
        momentum = float(prm.get("momentum", 0.9))
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(momentum, 0.999))

    def learn(self, train_data):
        """One pass over train_data; expects iterable of (images, captions)."""
        self.train()
        last_loss = 0.0
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)  # (B, T)

            # teacher forcing: input is up to T-1, predict next tokens
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            img_vec = self.encoder(images)             # (B, D)
            logits = self.decoder(img_vec, inputs)     # (B, T-1, V)

            loss = self.criteria[0](logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
            last_loss = float(loss.detach().cpu())
        return last_loss

    # Forward (training/inference)
    def forward(self, images, captions=None, hidden_state=None):
        """
        If captions is provided: returns (logits, None) where logits=(B, T-1, V).
        If captions is None: greedy decode, returns (generated_ids, None) where generated_ids=(B, L).
        """
        images = images.to(self.device)
        img_vec = self.encoder(images)  # (B, D)

        if captions is not None:
            captions = captions.to(self.device)
            inputs = captions[:, :-1]
            logits = self.decoder(img_vec, inputs)
            return logits, None

        # Inference: greedy generation
        B = images.size(0)
        cur = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)  # (B, 1)
        ended = torch.zeros(B, dtype=torch.bool, device=self.device)
        for _ in range(self.max_len - 1):
            logits = self.decoder(img_vec, cur)          # (B, t, V)
            next_logits = logits[:, -1, :]               # (B, V)
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            cur = torch.cat([cur, next_token], dim=1)    # (B, t+1)
            ended |= next_token.squeeze(1).eq(self.eos_idx)
            if ended.all():
                break
        return cur, None
