import torch
import torch.nn as nn


# -------------------- Positional Encoding (sinusoidal, batch-first) --------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [T,1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)  # [1,T,D] broadcast


# ------------------------------ Simple CNN encoder -> [B, N, D] ------------------------------

class TinyCNNEncoder(nn.Module):
    """Lightweight CNN that projects to embed_dim and returns region features [B, N, D]."""
    def __init__(self, in_channels: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, embed_dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> [B,D,H',W'] -> [B, N, D]
        x = self.body(x)
        B, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, D)
        return x  # memory for decoder


# ------------------------------ Transformer caption decoder ------------------------------

class MyTransformerDecoder(nn.Module):
    """Transformer decoder that consumes token embeddings and attends over image memory."""
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int = 768,
        hidden_size: int = 768,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(hidden_size)

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(self.d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(self.d_model, self.vocab_size)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # bool mask where True means "block"
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        inputs: torch.Tensor,          # [B, T] (teacher forcing) OR None (greedy)
        memory: torch.Tensor,          # [B, N, D]
        max_len: int,
        sos_token: int,
        eos_token: int,
    ) -> torch.Tensor:
        B = memory.size(0)
        device = memory.device

        if inputs is not None:
            # Teacher forcing
            assert inputs.dim() == 2, "inputs (captions) must be [B,T]"
            T = inputs.size(1)
            tgt = self.embedding(inputs)              # [B,T,D]
            tgt = self.pos_encoding(tgt)              # add positional enc
            tgt_mask = self._causal_mask(T, device)   # [T,T]
            dec = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)  # [B,T,D]
            logits = self.out_proj(dec)               # [B,T,V]
            return logits

        # Greedy inference
        ys = torch.full((B, 1), sos_token, dtype=torch.long, device=device)  # [B,1]
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len):
            T = ys.size(1)
            tgt = self.embedding(ys)                 # [B,T,D]
            tgt = self.pos_encoding(tgt)
            tgt_mask = self._causal_mask(T, device)
            dec = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)  # [B,T,D]
            next_logit = self.out_proj(dec[:, -1, :])                      # [B,V]
            next_tok = next_logit.argmax(dim=-1, keepdim=True)             # [B,1]
            ys = torch.cat([ys, next_tok], dim=1)                          # grow sequence
            finished = finished | (next_tok.squeeze(1) == eos_token)
            if finished.all():
                break
        return ys  # [B, <=max_len+1]


# ---------------------------------- Public API: Net ----------------------------------

class Net(nn.Module):
    """
    Image -> (encoder) -> memory [B,N,D]
    + captions (optional) -> (Transformer decoder)
    """
    def __init__(self, in_shape, out_shape, prm, device, *_, **__) -> None:
        super().__init__()
        self.device = device

        # Infer in_channels from in_shape ((C,H,W) or (N,C,H,W))
        def infer_in_channels(shape) -> int:
            if isinstance(shape, (tuple, list)):
                if len(shape) == 3:       # (C,H,W)
                    return int(shape[0])
                if len(shape) >= 4:       # (N,C,H,W)
                    return int(shape[1])
            return 3

        # Infer vocab_size from nested out_shape
        def flatten_ints(x):
            if isinstance(x, (tuple, list)):
                for xi in x:
                    for v in flatten_ints(xi):
                        yield v
            else:
                try:
                    yield int(x)
                except Exception:
                    return

        ints = list(flatten_ints(out_shape))
        if not ints:
            raise ValueError("out_shape must encode vocabulary size (e.g., ((V,),) or similar).")
        self.vocab_size = max(2, int(ints[0]))

        # Hyperparameters (hidden dim must be >= 640)
        self.embed_dim = max(int(prm.get("hidden_size", 768)), 640)
        self.max_len = int(prm.get("max_len", 20))
        self.sos_token = int(prm.get("sos_token", 1))
        self.eos_token = int(prm.get("eos_token", 2))
        self.pad_token = int(prm.get("pad_token", 0))

        in_channels = infer_in_channels(in_shape)

        self.encoder = TinyCNNEncoder(in_channels=in_channels, embed_dim=self.embed_dim)
        self.decoder = MyTransformerDecoder(
            vocab_size=self.vocab_size,
            feature_dim=self.embed_dim,
            hidden_size=self.embed_dim,
            num_layers=int(prm.get("num_layers", 2)),
            dropout=float(prm.get("dropout", 0.1)),
            num_heads=int(prm.get("num_heads", 8)),
        )

        self.criteria = None
        self.optimizer = None
        self.to(self.device)

    # ------------------------- training setup & loop -------------------------

    def train_setup(self, prm) -> None:
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.pad_token).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get("lr", 1e-3)),
            betas=(float(prm.get("momentum", 0.9)), 0.999),
        )

    def learn(self, train_data) -> None:
        """
        Optional training helper (teacher forcing).
        Expects iterable of (images, captions) with SOS/EOS in captions.
        """
        if self.optimizer is None or self.criteria is None:
            self.train_setup({})

        criterion = self.criteria[0]
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)          # [B,C,H,W]
            captions = captions.to(self.device)      # [B,T]

            # Teacher forcing: predict next token
            inp = captions[:, :-1]                   # [B,T-1]
            tgt = captions[:, 1:]                    # [B,T-1]

            logits, _ = self.forward(images, inp)    # [B,T-1,V]
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    # --------------------------------- forward ---------------------------------

    def forward(self, images: torch.Tensor, captions: torch.Tensor = None, hidden_state=None):
        """
        If captions is provided (teacher forcing):
            returns (logits[B,T-1,V], None)
        Else (greedy inference):
            returns (tokens[B,<=max_len+1], None)
        """
        assert images.dim() == 4, f"images must be [B,C,H,W], got {tuple(images.shape)}"
        images = images.to(self.device)
        memory = self.encoder(images)  # [B,N,D]

        if captions is not None:
            assert captions.dim() == 2, "captions must be [B,T]"
            captions = captions.to(self.device)
            logits = self.decoder(captions, memory, self.max_len, self.sos_token, self.eos_token)
            return logits, None

        tokens = self.decoder(None, memory, self.max_len, self.sos_token, self.eos_token)
        return tokens, None


def supported_hyperparameters():
    return {'lr', 'momentum'}
