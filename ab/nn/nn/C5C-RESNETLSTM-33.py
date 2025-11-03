import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {"lr", "momentum"}

# ---- small helpers ----
def _flatten_ints(x):
    """Flatten nested tuples/lists and yield ints inside."""
    if isinstance(x, (list, tuple)):
        for xi in x:
            yield from _flatten_ints(xi)
    else:
        try:
            yield int(x)
        except Exception:
            pass

def _infer_in_channels(in_shape):
    # Accept (C,H,W) or (N,C,H,W); default to 3 if ambiguous.
    if isinstance(in_shape, (list, tuple)):
        if len(in_shape) == 3:
            return int(in_shape[0])
        if len(in_shape) >= 4:
            return int(in_shape[1])
    return 3

# ---- model ----
class Net(nn.Module):
    """
    Minimal, robust CNN encoder.

    Modes:
      • Classification: out_shape -> (num_classes,)
      • Captioning: out_shape -> (vocab_size, max_len) (supports nested shapes)
    """
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm or {}

        # Figure out head type from out_shape
        ints = list(_flatten_ints(out_shape))
        if len(ints) <= 1:
            self.head = "class"
            self.num_classes = ints[0] if len(ints) == 1 else int(out_shape)
            self.vocab_size = None
            self.max_len = None
        else:
            self.head = "caption"
            self.vocab_size = int(ints[0])
            # default max_len from out_shape[1] if present, else prm or 16
            self.max_len = int(self.prm.get("max_len", ints[1]))
            self.num_classes = None

        in_channels = _infer_in_channels(in_shape)
        hidden = int(self.prm.get("hidden_dim", 256))
        embed = int(self.prm.get("embed_dim", 256))

        # ----- Encoder: small CNN -> global feature [B, hidden] -----
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, hidden),
        )

        if self.head == "class":
            self.head_layer = nn.Linear(hidden, self.num_classes)
        else:
            # ----- Captioning decoder -----
            self.embedding = nn.Embedding(self.vocab_size, embed)
            # feed [embed + image_feature] to LSTM
            self.lstm = nn.LSTM(input_size=embed + hidden, hidden_size=hidden, num_layers=1, batch_first=True)
            self.proj = nn.Linear(hidden, self.vocab_size)
            self.sos_token = int(self.prm.get("sos_token", 1))
            self.eos_token = int(self.prm.get("eos_token", 2))

        # training state
        self.criterion = None
        self.criteria = None
        self.optimizer = None

        self.to(self.device)

    # ---- training hooks ----
    def train_setup(self, prm):
        prm = prm or {}
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.to(self.device)
        if self.head == "class":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            # assume PAD=0
            self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.criteria = (self.criterion,)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        if self.optimizer is None or self.criterion is None:
            self.train_setup(self.prm)
        self.train()
        for batch in train_data:
            if self.head == "class":
                images, targets = batch
                images, targets = images.to(self.device), targets.to(self.device)
                logits = self.forward(images)  # [B, C]
                loss = self.criterion(logits, targets)
            else:
                images, captions = batch  # captions [B, T]
                images, captions = images.to(self.device), captions.to(self.device)
                logits, tgt = self.forward(images, captions)  # logits [B, T-1, V], tgt [B, T-1]
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # ---- forward ----
    def forward(self, images, captions=None, hidden_state=None):
        images = images.to(self.device)
        feats = self.encoder(images)  # [B, hidden]

        if self.head == "class":
            return self.head_layer(feats)  # [B, num_classes]

        # Captioning path
        B = feats.size(0)
        if captions is not None:
            # Teacher forcing: predict next tokens for input tokens (shifted)
            captions = captions.to(self.device)
            # input excludes last token; target excludes first token
            inp = captions[:, :-1]
            tgt = captions[:, 1:]

            emb = self.embedding(inp)                              # [B, T-1, E]
            ctx = feats.unsqueeze(1).expand(B, emb.size(1), -1)    # [B, T-1, H]
            dec_in = torch.cat([emb, ctx], dim=-1)                 # [B, T-1, E+H]
            out, hidden_state = self.lstm(dec_in, hidden_state)    # [B, T-1, H]
            logits = self.proj(out)                                # [B, T-1, V]
            return logits, tgt

        # Greedy decoding for inference
        max_len = int(self.max_len or 16)
        inputs = torch.full((B, 1), self.sos_token, dtype=torch.long, device=self.device)
        h = hidden_state
        outputs = []
        for _ in range(max_len):
            emb = self.embedding(inputs[:, -1:])   # [B,1,E]
            ctx = feats.unsqueeze(1)               # [B,1,H]
            dec_in = torch.cat([emb, ctx], dim=-1) # [B,1,E+H]
            out, h = self.lstm(dec_in, h)          # [B,1,H]
            logit = self.proj(out)                 # [B,1,V]
            next_tok = logit.argmax(dim=-1)        # [B,1]
            outputs.append(next_tok)
            inputs = torch.cat([inputs, next_tok], dim=1)
            if (next_tok == self.eos_token).all():
                break

        return torch.cat(outputs, dim=1), h
