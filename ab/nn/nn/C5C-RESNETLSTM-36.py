import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional


def supported_hyperparameters() -> set:
    return {"lr", "momentum"}


# ------------------------------- Utilities -------------------------------

def _flatten_ints(x):
    """Robustly extract integers (e.g., vocab size) from nested tuples/lists."""
    if isinstance(x, (tuple, list)):
        for xi in x:
            yield from _flatten_ints(xi)
    else:
        try:
            yield int(x)
        except Exception:
            pass


def _infer_in_channels(in_shape) -> int:
    """
    Accepts shapes like (C,H,W) or (N,C,H,W). Falls back to 3.
    """
    if isinstance(in_shape, (tuple, list)):
        if len(in_shape) == 3:
            return int(in_shape[0])
        if len(in_shape) >= 4:
            return int(in_shape[1])
    return 3


# -------------------------- ResNet-ish Spatial Encoder --------------------------

class ResNetSpatialEncoder(nn.Module):
    """
    Lightweight conv encoder that outputs region features [B, N, D]
    suitable for spatial attention. No torchvision dependency.
    """
    def __init__(self, in_channels: int = 3, output_dim: int = 768) -> None:
        super().__init__()
        c = in_channels
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 3 downsampling stages (factor 8 total) -> decent region grid
        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.stage1 = block(64, 128, stride=2)   # /4
        self.stage2 = block(128, 256, stride=2)  # /8
        self.stage3 = block(256, 256, stride=1)  # keep /8

        # 1x1 projection to desired feature dim
        self.proj = nn.Conv2d(256, output_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)              # [B, 256, H', W']
        x = self.proj(x)                # [B, D, H', W']
        B, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, D)  # [B, N, D]
        return x


# -------------------------- Spatial-Attention LSTM Decoder --------------------------

class SpatialAttentionLSTMDecoder(nn.Module):
    """
    Decoder with additive attention over spatial regions.
    - features: [B, N, D] (N regions, D feature dim)
    - teacher forcing: captions_in [B, T_in] -> logits [B, T_in, V]
    - greedy inference when captions_in is None
    """
    def __init__(self, vocab_size: int, feature_dim: int, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.feature_dim = int(feature_dim)
        self.hidden_size = int(hidden_size)

        # Use hidden_size for token embeddings
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)

        # Attention: additive (Bahdanau-style)
        self.attn_feat = nn.Linear(self.feature_dim, self.hidden_size, bias=False)
        self.attn_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.attn_score = nn.Linear(self.hidden_size, 1, bias=False)

        # LSTMCell inputs: [emb_t; context] where context \in R^{feature_dim}
        self.lstm_cell = nn.LSTMCell(self.hidden_size + self.feature_dim, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def init_zero_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(batch_size, self.hidden_size, device=device)
        return h0, c0

    def _attend(self, features: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        features: [B, N, D], h: [B, H] -> context: [B, D]
        """
        # Compute attention scores over N regions
        proj_feat = self.attn_feat(features)                 # [B, N, H]
        proj_h = self.attn_hidden(h).unsqueeze(1)            # [B, 1, H]
        scores = self.attn_score(torch.tanh(proj_feat + proj_h)).squeeze(-1)  # [B, N]
        alpha = torch.softmax(scores, dim=1)                 # [B, N]
        context = (alpha.unsqueeze(-1) * features).sum(dim=1)  # [B, D]
        return context

    def forward(
        self,
        features: torch.Tensor,                # [B, N, D]
        captions_in: Optional[torch.Tensor],   # [B, T_in] or None
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        sos_token: int,
        eos_token: int,
        max_len: int,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B = features.size(0)
        device = features.device

        # init hidden if needed
        if hidden_state is None:
            h, c = self.init_zero_hidden(B, device)
        else:
            h, c = hidden_state

        if captions_in is not None:
            # Teacher forcing
            T = captions_in.size(1)
            logits_collect = []
            for t in range(T):
                emb_t = self.embed(captions_in[:, t])        # [B, H]
                ctx = self._attend(features, h)              # [B, D]
                lstm_in = torch.cat([emb_t, ctx], dim=-1)    # [B, H+D]
                h, c = self.lstm_cell(lstm_in, (h, c))       # each [B, H]
                h = self.dropout(h)
                logits_collect.append(self.out(h).unsqueeze(1))  # [B,1,V]
            logits = torch.cat(logits_collect, dim=1)         # [B,T,V]
            return logits, (h, c)

            # (Note: caller should time-shift inputs/targets if needed)
        else:
            # Greedy decoding
            tokens = []
            cur = torch.full((B,), sos_token, dtype=torch.long, device=device)
            for _ in range(max_len):
                emb_t = self.embed(cur)                      # [B, H]
                ctx = self._attend(features, h)              # [B, D]
                lstm_in = torch.cat([emb_t, ctx], dim=-1)    # [B, H+D]
                h, c = self.lstm_cell(lstm_in, (h, c))
                h = self.dropout(h)
                logit = self.out(h)                          # [B, V]
                cur = logit.argmax(dim=-1)                   # [B]
                tokens.append(cur.unsqueeze(1))              # [B,1]
                if (cur == eos_token).all():
                    break
            return torch.cat(tokens, dim=1), (h, c)


# ------------------------------------ Net ------------------------------------

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device

        # derive vocab size robustly from out_shape
        ints = list(_flatten_ints(out_shape))
        if not ints:
            raise ValueError("out_shape must encode at least the vocab size.")
        self.vocab_size = max(2, ints[0])

        # config
        self.hidden_size = int(prm.get("hidden_size", 768))  # ≥640
        self.max_len = int(prm.get("max_len", 20))
        self.sos_token = int(prm.get("sos_token", 1))
        self.eos_token = int(prm.get("eos_token", 2))
        self.pad_token = int(prm.get("pad_token", 0))

        in_channels = _infer_in_channels(in_shape)
        self.cnn = ResNetSpatialEncoder(in_channels=in_channels, output_dim=self.hidden_size)
        self.rnn = SpatialAttentionLSTMDecoder(
            vocab_size=self.vocab_size,
            feature_dim=self.hidden_size,
            hidden_size=self.hidden_size,
        )

        # training state
        self.criteria = None
        self.criterion = None
        self.optimizer = None

        self.to(self.device)

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token).to(self.device)
        self.criteria = (self.criterion,)
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=float(prm.get("lr", 1e-3)),
            betas=(float(prm.get("momentum", 0.9)), 0.999),
        )

    def learn(self, train_data) -> None:
        """
        Expects batches of (images, captions) where captions are [B,T] and include SOS/targets.
        """
        if self.optimizer is None or self.criterion is None:
            self.train_setup({})

        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)

            # teacher forcing: predict next token for each position except the last
            inp = captions[:, :-1]   # [B, T-1]
            tgt = captions[:, 1:]    # [B, T-1]

            features = self.cnn(images)  # [B, N, D]
            logits, _ = self.rnn(features, inp, None, self.sos_token, self.eos_token, self.max_len)  # [B,T-1,V]

            loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        - If captions is provided (teacher forcing):
            returns (logits[B,T-1,V], targets[B,T-1])
        - Else (inference):
            returns (tokens[B,<=max_len], hidden_state)
        """
        assert images.dim() == 4, f"images must be [B,C,H,W], got {images.shape}"
        features = self.cnn(images.to(self.device))  # [B, N, D]

        if captions is not None:
            captions = captions.to(self.device)
            assert captions.dim() == 2, "captions must be [B,T]"
            inp = captions[:, :-1]
            tgt = captions[:, 1:]
            logits, hidden_state = self.rnn(features, inp, hidden_state, self.sos_token, self.eos_token, self.max_len)
            # Shape checks
            assert logits.shape[:2] == tgt.shape[:2], "time dimension mismatch between logits and targets"
            return logits, tgt
        else:
            tokens, hidden_state = self.rnn(features, None, hidden_state, self.sos_token, self.eos_token, self.max_len)
            return tokens, hidden_state
