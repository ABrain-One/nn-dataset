import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


# ----------------------------- SE + (simple) CBAM block -----------------------------

class ConvBlockSECBAM(nn.Module):
    """
    Conv -> BN -> ReLU -> DW-Conv -> BN, then SE and simple channel/spatial attentions.
    Uses residual when shapes match; otherwise projects the residual.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.dwconv = nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding, stride=1, groups=out_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        # SE: squeeze (avg pool) -> FC -> ReLU -> FC -> Sigmoid
        hidden = max(8, out_ch // max(1, reduction))
        self.se_fc1 = nn.Linear(out_ch, hidden)
        self.se_fc2 = nn.Linear(hidden, out_ch)

        # Channel attention (MLP with softmax across channels)
        self.ca_fc1 = nn.Linear(out_ch, hidden)
        self.ca_fc2 = nn.Linear(hidden, out_ch)

        # Spatial attention via pooled descriptors -> MLP -> sigmoid
        self.sa_fc1 = nn.Linear(out_ch, hidden)
        self.sa_fc2 = nn.Linear(hidden, out_ch)

        # Residual projection if needed
        self.proj = None
        if in_ch != out_ch or stride != 1:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        y = self.act(self.bn1(self.conv1(x)))
        y = self.act(self.bn2(self.dwconv(y)))  # [B, C, H, W]

        # ----- SE -----
        b, c, h, w = y.shape
        se_vec = y.mean(dim=(2, 3))                 # [B, C]
        se_vec = F.relu(self.se_fc1(se_vec), inplace=True)
        se_vec = torch.sigmoid(self.se_fc2(se_vec)) # [B, C]
        y = y * se_vec.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W]

        # ----- Channel attention -----
        ch_vec = y.mean(dim=(2, 3))                 # [B, C]
        ch_vec = F.relu(self.ca_fc1(ch_vec), inplace=True)
        ch_vec = F.softmax(self.ca_fc2(ch_vec), dim=1)  # [B, C], soft selection
        y = y * ch_vec.unsqueeze(-1).unsqueeze(-1)

        # ----- Spatial attention (lightweight) -----
        # pool -> [B, C] -> weights [B, C] -> per-channel gating (broadcast spatially)
        sp_vec = F.adaptive_max_pool2d(y, (1, 1)).view(b, c)  # [B, C]
        sp_vec = F.relu(self.sa_fc1(sp_vec), inplace=True)
        sp_vec = torch.sigmoid(self.sa_fc2(sp_vec))           # [B, C]
        y = y * sp_vec.unsqueeze(-1).unsqueeze(-1)

        # Residual
        if self.proj is not None:
            residual = self.proj(residual)
        return y + residual


# ----------------------------- Feature fusion (1x1 convs) -----------------------------

class FeatureFusion(nn.Module):
    def __init__(self, feature_dim, dropout=0.1):
        super().__init__()
        self.fcn1 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fcn2 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.fcn2(self.fcn1(x))


# ----------------------------- Positional encoding (sinusoidal) -----------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [T,1]
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)


# ----------------------------- Transformer caption decoder -----------------------------

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos = SinusoidalPositionalEncoding(self.d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out = nn.Linear(self.d_model, self.vocab_size)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # True = block
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, inputs, memory, max_len, sos_token, eos_token):
        """
        inputs: [B, T] (teacher forcing) or None (greedy)
        memory: [B, N, D] from encoder
        returns:
          - if teacher forcing: logits [B, T, V]
          - if greedy: tokens [B, <=max_len+1]
        """
        device = memory.device
        if inputs is not None:
            assert inputs.dim() == 2, "captions must be [B,T]"
            B, T = inputs.shape
            tgt = self.embedding(inputs)          # [B,T,D]
            tgt = self.pos(tgt)
            tgt_mask = self._causal_mask(T, device)
            dec = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            return self.out(dec)

        # Greedy generation
        B = memory.size(0)
        ys = torch.full((B, 1), sos_token, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len):
            T = ys.size(1)
            tgt = self.embedding(ys)
            tgt = self.pos(tgt)
            tgt_mask = self._causal_mask(T, device)
            dec = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            next_logit = self.out(dec[:, -1, :])      # [B,V]
            next_tok = next_logit.argmax(dim=-1, keepdim=True)  # [B,1]
            ys = torch.cat([ys, next_tok], dim=1)
            finished |= (next_tok.squeeze(1) == eos_token)
            if finished.all():
                break
        return ys


# ----------------------------------- Encoder -----------------------------------

class TinySECBAMEncoder(nn.Module):
    """
    A tiny CNN encoder that returns region features [B, N, D].
    """
    def __init__(self, in_ch: int = 3, embed_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        mid = max(64, embed_dim // 8)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid, mid, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        self.block1 = ConvBlockSECBAM(mid, mid, stride=1)
        self.block2 = ConvBlockSECBAM(mid, mid, stride=1)
        self.block3 = ConvBlockSECBAM(mid, embed_dim, stride=2)  # up to D

        self.fuse = FeatureFusion(embed_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fuse(x)                      # [B, D, H, W]
        B, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, D)  # [B, N, D]
        return x


# ------------------------------------- Net -------------------------------------

class Net(nn.Module):
    """
    Images -> encoder -> memory [B,N,D]
    + (optional) captions -> transformer decoder
    """
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device

        # infer input channels from in_shape ((C,H,W) or (N,C,H,W))
        def infer_in_channels(shape) -> int:
            if isinstance(shape, (tuple, list)):
                if len(shape) >= 4:
                    return int(shape[1])
                if len(shape) == 3:
                    return int(shape[0])
            return 3

        # infer vocab size from nested out_shape
        def flatten_ints(x):
            if isinstance(x, (tuple, list)):
                for xi in x:
                    yield from flatten_ints(xi)
            else:
                try:
                    yield int(x)
                except Exception:
                    return

        ints = list(flatten_ints(out_shape))
        if not ints:
            raise ValueError("out_shape must encode vocabulary size (e.g., ((V,),) or similar).")
        self.vocab_size = max(2, int(ints[0]))

        # hyperparams
        self.embed_dim = int(prm.get("hidden_size", 768))
        self.max_len = int(prm.get("max_len", 20))
        self.sos_token = int(prm.get("sos_token", 1))
        self.eos_token = int(prm.get("eos_token", 2))
        self.pad_token = int(prm.get("pad_token", 0))
        self.dropout = float(prm.get("dropout", 0.1))
        self.num_layers = int(prm.get("num_layers", 2))
        self.nhead = int(prm.get("num_heads", 8))

        in_ch = infer_in_channels(in_shape)
        self.encoder = TinySECBAMEncoder(in_ch=in_ch, embed_dim=self.embed_dim, dropout=self.dropout)
        self.decoder = TransformerDecoder(
            vocab_size=self.vocab_size,
            d_model=self.embed_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        self.criteria = None
        self.optimizer = None
        self.to(self.device)

    # ------------------- training setup / loop (optional helper) -------------------

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.pad_token).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get("lr", 1e-3)),
            betas=(float(prm.get("momentum", 0.9)), 0.999),
        )

    def learn(self, train_data):
        """
        Expects iterable of (images, captions) with SOS/EOS already inserted.
        Uses teacher forcing to predict next token.
        """
        if self.optimizer is None or self.criteria is None:
            self.train_setup({})

        criterion = self.criteria[0]
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)         # [B,C,H,W]
            captions = captions.to(self.device)     # [B,T]

            # teacher forcing: shift by 1
            inp = captions[:, :-1]                  # [B,T-1]
            tgt = captions[:, 1:]                   # [B,T-1]

            logits, _ = self.forward(images, inp)   # [B,T-1,V]
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    # ------------------------------------ forward ------------------------------------

    def forward(self, images: torch.Tensor, captions: torch.Tensor = None, hidden_state=None):
        """
        If captions provided (teacher forcing): returns (logits[B,T,V], None)
        Else: returns (tokens[B,<=max_len+1], None) via greedy decoding.
        """
        assert images.dim() == 4, f"images must be [B,C,H,W], got {tuple(images.shape)}"
        images = images.to(self.device)
        memory = self.encoder(images)  # [B,N,D]

        if captions is not None:
            captions = captions.to(self.device)
            logits = self.decoder(captions, memory, self.max_len, self.sos_token, self.eos_token)
            return logits, None

        tokens = self.decoder(None, memory, self.max_len, self.sos_token, self.eos_token)
        return tokens, None
