import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


# ---------- simple CNN encoder -> global feature ----------
class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            # stem
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # downsample 1
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # downsample 2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # [B, 256, 1, 1]
            nn.Flatten(),             # [B, 256]
            nn.Linear(256, hidden_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, hidden]


# ---------- LSTM decoder (concat embedding + image context) ----------
class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed = nn.Embedding(self.vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim + hidden_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True)
        self.proj = nn.Linear(hidden_size, self.vocab_size)

    def init_zero_hidden(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        return (h0, c0)

    def forward(self,
                captions_in: torch.Tensor,  # [B, T_in] tokens (teacher forcing inputs)
                img_context: torch.Tensor,  # [B, H] image feature
                hidden: tuple | None = None) -> tuple[torch.Tensor, tuple]:
        """
        Returns logits: [B, T_in, V]
        """
        B, T = captions_in.shape
        emb = self.embed(captions_in)                      # [B, T, E]
        ctx = img_context.unsqueeze(1).expand(B, T, -1)    # [B, T, H]
        dec_in = torch.cat([emb, ctx], dim=-1)             # [B, T, E+H]
        out, hidden = self.lstm(dec_in, hidden)            # [B, T, H], (h,c)
        logits = self.proj(out)                            # [B, T, V]
        return logits, hidden


# ---------- utility to robustly extract ints from weird out_shape nests ----------
def _flatten_ints(x):
    if isinstance(x, (list, tuple)):
        for xi in x:
            yield from _flatten_ints(xi)
    else:
        try:
            yield int(x)
        except Exception:
            pass


def _infer_in_channels(in_shape) -> int:
    # supports (C,H,W) or (N,C,H,W)
    if isinstance(in_shape, (list, tuple)):
        if len(in_shape) == 3:
            return int(in_shape[0])
        if len(in_shape) >= 4:
            return int(in_shape[1])
    return 3


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        """
        Forward signatures:
          - Training (teacher forcing): forward(images, captions) -> (logits[B,T-1,V], targets[B,T-1])
          - Inference: forward(images) -> (tokens[B,<=max_len], hidden)
        """
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm or {}

        # hyperparams / sizes
        self.hidden_size = int(self.prm.get("hidden_size", 768))  # ≥640 good default
        self.embed_dim = int(self.prm.get("embed_dim", 256))
        self.max_len = int(self.prm.get("max_len", 16))
        self.pad_token = int(self.prm.get("pad_token", 0))
        self.sos_token = int(self.prm.get("sos_token", 1))
        self.eos_token = int(self.prm.get("eos_token", 2))

        # Determine vocab_size from out_shape robustly
        ints = list(_flatten_ints(out_shape))
        if len(ints) == 0:
            raise ValueError("out_shape must encode at least the vocab size.")
        self.vocab_size = max(2, ints[0])

        # Encoder / Decoder
        in_channels = _infer_in_channels(in_shape)
        self.encoder = CNNEncoder(in_channels=in_channels, hidden_size=self.hidden_size)
        self.decoder = LSTMDecoder(vocab_size=self.vocab_size,
                                   embed_dim=self.embed_dim,
                                   hidden_size=self.hidden_size)

        # training setup placeholders
        self.criterion = None
        self.criteria = None
        self.optimizer = None

        self.to(self.device)

    # ---- training helpers ----
    def train_setup(self, prm):
        prm = prm or {}
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.to(self.device)
        # PAD=0 by convention
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token).to(self.device)
        self.criteria = (self.criterion,)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        """
        Expects batches of (images, captions) with captions LongTensor [B,T] including SOS/targets.
        """
        if self.optimizer is None or self.criterion is None:
            self.train_setup(self.prm)

        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)

            logits, targets = self.forward(images, captions)  # [B,T-1,V], [B,T-1]
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    # ---- forward ----
    def forward(self, images: torch.Tensor, captions: torch.Tensor | None = None, hidden_state: tuple | None = None):
        """
        images:   [B,C,H,W]
        captions: [B,T] (teacher forcing). If provided, we train to predict next token.
        returns:
          - if captions is not None: (logits[B,T-1,V], targets[B,T-1])
          - else: (generated_tokens[B,<=max_len], hidden_state)
        """
        assert images.dim() == 4, f"images must be [B,C,H,W], got {images.dim()}D"

        # encode image to global feature
        feats = self.encoder(images.to(self.device))  # [B, H]
        B = feats.size(0)

        if captions is not None:
            assert captions.dim() == 2 and captions.size(0) == B, "captions must be [B,T]"
            captions = captions.to(self.device)

            # teacher forcing: input = captions[:, :-1], target = captions[:, 1:]
            inp = captions[:, :-1]
            tgt = captions[:, 1:]

            logits, hidden_state = self.decoder(inp, feats, hidden_state)  # [B,T-1,V]
            # shape checks
            assert logits.shape[:2] == tgt.shape[:2], "time dimension mismatch between logits and targets"
            return logits, tgt

        # inference: greedy decode until EOS or max_len
        tokens = torch.full((B, 1), self.sos_token, dtype=torch.long, device=self.device)  # [B,1]
        h = hidden_state
        generated = []

        for _ in range(self.max_len):
            logits, h = self.decoder(tokens[:, -1:].contiguous(), feats, h)  # step on last token; logits [B,1,V]
            next_tok = logits.argmax(dim=-1)  # [B,1]
            generated.append(next_tok)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == self.eos_token).all():
                break

        return torch.cat(generated, dim=1), h
