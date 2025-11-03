import torch
import torch.nn as nn

def supported_hyperparameters():
    return {"lr", "momentum"}

# --------- small helpers ----------
def _flatten_ints(x):
    """Yield all ints found inside nested tuples/lists (robust for odd out_shape layouts)."""
    if isinstance(x, (list, tuple)):
        for xi in x:
            yield from _flatten_ints(xi)
    else:
        try:
            yield int(x)
        except Exception:
            pass

def _infer_in_channels(in_shape):
    """
    Accepts (C,H,W) or (N,C,H,W); defaults to 3 if ambiguous.
    """
    if isinstance(in_shape, (tuple, list)):
        if len(in_shape) == 3:
            return int(in_shape[0])
        if len(in_shape) >= 4:
            return int(in_shape[1])
    return 3


class Net(nn.Module):
    """
    Minimal, robust CNN encoder + (optional) caption LSTM decoder.

    Modes:
      • Classification when out_shape encodes a single integer -> logits [B, num_classes]
      • Captioning when out_shape encodes at least two ints (e.g., vocab, max_len)
        -> training: returns (logits[B,T-1,V], targets[B,T-1])
        -> inference: returns (tokens[B,≤max_len], hidden)
    """
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm or {}

        # Determine task type from out_shape
        ints = list(_flatten_ints(out_shape))
        if len(ints) <= 1:
            # classification
            self.head_type = "class"
            self.num_classes = (ints[0] if len(ints) == 1 else int(out_shape)) or 1
            self.vocab_size = None
            self.max_len = None
        else:
            # captioning
            self.head_type = "caption"
            self.vocab_size = max(2, int(ints[0]))
            # if a length is present, use it; else default
            self.max_len = int(self.prm.get("max_len", (ints[1] if len(ints) > 1 else 16)))
            self.num_classes = None

        in_channels = _infer_in_channels(in_shape)
        hidden = int(self.prm.get("hidden_dim", 768))  # ≥640 as a safe default
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

        if self.head_type == "class":
            self.classifier = nn.Linear(hidden, self.num_classes)
        else:
            # ----- Captioning decoder -----
            self.embedding = nn.Embedding(self.vocab_size, embed)
            # LSTM takes [embed + image_feature] at each step
            self.decoder = nn.LSTM(input_size=embed + hidden, hidden_size=hidden, num_layers=1, batch_first=True)
            self.proj = nn.Linear(hidden, self.vocab_size)
            self.sos_token = int(self.prm.get("sos_token", 1))  # PAD=0, SOS=1, EOS=2 by default
            self.eos_token = int(self.prm.get("eos_token", 2))

        # training state (filled by train_setup)
        self.criterion = None
        self.criteria = None
        self.optimizer = None

        self.to(self.device)

    # ---------- training utilities ----------
    def train_setup(self, prm):
        prm = prm or {}
        lr = prm.get("lr", None)
        # Keep the expected AdamW signature (momentum used as beta1)
        beta1 = float(prm.get("momentum", 0.9))
        self.to(self.device)
        if self.head_type == "class":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            # Assume PAD=0
            self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.criteria = (self.criterion,)
        # lr is required by the surrounding pipeline in previous epochs
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        """
        Generic training loop. Expects:
          • classification: batches of (images, labels)
          • captioning:     batches of (images, captions) with PAD=0 and SOS/EOS tokens
        """
        if self.optimizer is None or self.criterion is None:
            self.train_setup(self.prm)

        self.train()
        for batch in train_data:
            if self.head_type == "class":
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.forward(images)  # [B, C]
                loss = self.criterion(logits, labels)
            else:
                images, captions = batch  # captions [B, T] (long)
                images = images.to(self.device)
                captions = captions.to(self.device)
                logits, targets = self.forward(images, captions)  # [B,T-1,V], [B,T-1]
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    # ---------- forward ----------
    def forward(self, images, captions=None, hidden_state=None):
        """
        images:  [B, C, H, W]
        captions (optional): [B, T] LongTensor with PAD=0 (teacher forcing if provided)
        returns:
          • classification: logits [B, num_classes]
          • captioning:
              training:  (logits[B,T-1,V], targets[B,T-1])
              inference: (tokens[B,≤max_len], hidden_state)
        """
        # Basic shape checks for images
        assert images.dim() == 4, f"images must be 4D [B,C,H,W], got {images.dim()}D"
        feats = self.encoder(images.to(self.device))  # [B, hidden]

        if self.head_type == "class":
            return self.classifier(feats)

        # ---- captioning ----
        B = feats.size(0)

        if captions is not None:
            # teacher forcing: predict next token for each input token (shifted)
            assert captions.dim() == 2, f"captions must be 2D [B,T], got {captions.dim()}D"
            assert captions.size(0) == B, "batch size mismatch between images and captions"
            captions = captions.to(self.device)

            # input excludes last token; target excludes first token
            inp = captions[:, :-1]             # [B, T-1]
            tgt = captions[:, 1:]              # [B, T-1]

            emb = self.embedding(inp)          # [B, T-1, E]
            ctx = feats.unsqueeze(1).expand(B, emb.size(1), feats.size(1))  # [B, T-1, H]
            dec_in = torch.cat([emb, ctx], dim=-1)                          # [B, T-1, E+H]

            out, hidden_state = self.decoder(dec_in, hidden_state)          # [B, T-1, H]
            logits = self.proj(out)                                         # [B, T-1, V]
            return logits, tgt

        # Greedy decode for inference
        max_len = int(self.max_len or 16)
        tokens = torch.full((B, 1), self.sos_token, dtype=torch.long, device=self.device)  # [B,1]
        h = hidden_state
        outputs = []

        for _ in range(max_len):
            emb = self.embedding(tokens[:, -1:])          # [B,1,E]
            ctx = feats.unsqueeze(1)                       # [B,1,H]
            dec_in = torch.cat([emb, ctx], dim=-1)         # [B,1,E+H]
            out, h = self.decoder(dec_in, h)               # [B,1,H]
            logit = self.proj(out)                         # [B,1,V]
            next_tok = logit.argmax(dim=-1)                # [B,1]
            outputs.append(next_tok)
            tokens = torch.cat([tokens, next_tok], dim=1)  # grow sequence
            # early stop if all hit EOS
            if (next_tok == self.eos_token).all():
                break

        return torch.cat(outputs, dim=1), h
