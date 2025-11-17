import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple


def supported_hyperparameters():
    return {"lr", "momentum"}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: Dict[str, Any], device: torch.device):
        super().__init__()
        self.device = device
        self.prm = prm or {}

        # ---- basic config ----
        self.grad_clip_value = float(self.prm.get("grad_clip", 3.0))
        self.dropout_p = float(self.prm.get("dropout", 0.1))
        self.hidden_size = int(self.prm.get("hidden_size", 768))
        self.num_layers = int(self.prm.get("num_layers", 1))
        self.decoder_type = self.prm.get("decoder_type", "lstm")
        self.sos_idx = int(self.prm.get("sos_idx", 1))
        self.pad_idx = int(self.prm.get("pad_idx", 0))
        self.eos_idx = int(self.prm.get("eos_idx", 2))
        self.max_decode_len = int(self.prm.get("max_decode_len", 16))

        # ---- in/out shapes ----
        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # ---- backbone (torchvision-free) ----
        backbone_type = str(self.prm.get("backbone", "resnet")).lower()
        if backbone_type == "mobilenetv3":
            self.spatial_encoder = nn.Sequential(
                self._make_fire_layer(self.in_channels, 64, float(self.prm.get("width_mult", 1.0))),
                self._make_fire_layer(64, 128, float(self.prm.get("width_mult", 1.0))),
            )
            enc_out_channels = 128
        elif backbone_type == "efficientnet":
            self.spatial_encoder = nn.Sequential(
                nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            enc_out_channels = 128
        else:
            self.spatial_encoder = nn.Sequential(
                nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            enc_out_channels = 128

        self.feature_dim = enc_out_channels

        # ---- decoder ----
        self.decoder = self.build_decoder(self.hidden_size, self.num_layers)

        # aliases some harness utils may inspect
        if hasattr(self.decoder, "embed"):
            self.embedding = self.decoder.embed
        if hasattr(self.decoder, "proj_to_vocab"):
            self.fc_out = self.decoder.proj_to_vocab

        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    # ---------------- API helpers ----------------
    @staticmethod
    def supported_hyperparameters():
        return {"lr", "momentum"}

    @staticmethod
    def _infer_in_channels(in_shape: Any) -> int:
        # Handles (C,H,W) or (N,C,H,W); default 3
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) == 3 and all(isinstance(v, int) for v in in_shape):
                return int(in_shape[0])           # (C,H,W)
            if len(in_shape) >= 2 and isinstance(in_shape[1], int):
                return int(in_shape[1])           # (N,C,H,W)
        return 3

    @staticmethod
    def _first_int(x: Any) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            for item in x:
                try:
                    return Net._first_int(item)
                except Exception:
                    continue
        return int(x)

    def _normalize_captions(self, captions: torch.Tensor) -> torch.Tensor:
        if captions.dim() == 1:
            captions = captions.unsqueeze(0)
        elif captions.dim() == 3 and captions.size(1) == 1:
            captions = captions[:, 0, :]
        return captions

    # ---------------- training setup ----------------
    def train_setup(self, prm):
        prm = prm or {}
        self.to(self.device)
        self.train()
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device)
        self.criteria = (self.criterion,)
        lr = float(prm.get("lr", self.prm.get("lr", 1e-3)))
        beta1 = float(prm.get("momentum", self.prm.get("momentum", 0.9)))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data: Any):
        # robust unpacking; works with tuples, dicts, or objects with .x/.y
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

            if captions.dim() != 2 or captions.size(1) <= 1:
                continue

            inputs = captions[:, :-1]   # [B, T-1]
            targets = captions[:, 1:]   # [B, T-1]

            self.optimizer.zero_grad(set_to_none=True)
            logits, _ = self.forward(images, inputs)  # [B, T-1, V]

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_value)
            self.optimizer.step()

    # ---------------- blocks ----------------
    def _make_fire_layer(self, in_channels: int, out_channels: int, width_mult: float) -> nn.Sequential:
        mid = max(16, int(16 * width_mult))
        outc = int(out_channels)
        return nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, outc, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )

    def build_decoder(self, hidden_size: int, num_layers: int) -> nn.Module:
        decoder_type = getattr(self, "decoder_type", "lstm").lower()

        if decoder_type == "transformer":
            class CaptionTransformerDecoder(nn.Module):
                def __init__(self, d_model: int, num_heads: int, vocab_size: int, pad_idx: int, dropout: float):
                    super().__init__()
                    self.hidden_size = d_model
                    self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
                    layer = nn.TransformerDecoderLayer(
                        d_model=d_model,
                        nhead=num_heads,
                        dim_feedforward=min(3072, d_model * 4),
                        dropout=dropout,
                        batch_first=True,
                    )
                    self.dec = nn.TransformerDecoder(layer, num_layers=2)
                    self.proj = nn.Linear(d_model, vocab_size)

                def _causal_mask(self, T: int, device: torch.device):
                    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), 1)

                def forward(self, features: torch.Tensor, inputs: torch.Tensor, hidden_state: Optional[Tuple] = None):
                    emb = self.embed(inputs)  # [B, T, H]
                    tgt_mask = self._causal_mask(emb.size(1), emb.device)
                    out = self.dec(tgt=emb, memory=features, tgt_mask=tgt_mask)  # [B, T, H]
                    logits = self.proj(out)  # [B, T, V]
                    return logits, None

                def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
                    z = torch.zeros(batch, self.hidden_size, device=device)
                    return z, z

            dec = CaptionTransformerDecoder(
                d_model=self.hidden_size,
                num_heads=8,
                vocab_size=self.vocab_size,
                pad_idx=self.pad_idx,
                dropout=self.dropout_p,
            )
            setattr(self, "rnn", dec)
            return dec

        class LSTMDecoder(nn.Module):
            def __init__(self, vocab_size: int, hidden_size: int, pad_idx: int):
                super().__init__()
                self.hidden_size = hidden_size
                self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
                self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)
                self.proj_to_vocab = nn.Linear(hidden_size, vocab_size)

            def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
                return (
                    torch.zeros(batch, self.hidden_size, device=device),
                    torch.zeros(batch, self.hidden_size, device=device),
                )

            def forward(
                self,
                features: torch.Tensor,
                inputs: torch.Tensor,
                hidden_state: Optional[Tuple] = None,
            ) -> Tuple[torch.Tensor, Tuple]:
                B, T = inputs.size(0), inputs.size(1)
                h, c = (self.init_zero_hidden(B, inputs.device) if hidden_state is None else hidden_state)
                outputs = []
                for t in range(T):
                    token_t = inputs[:, t]
                    emb_t = self.embed(token_t)
                    h, c = self.lstm_cell(emb_t, (h, c))
                    logits_t = self.proj_to_vocab(h)
                    outputs.append(logits_t)
                logits = torch.stack(outputs, dim=1)
                return logits, (h, c)

        dec = LSTMDecoder(self.vocab_size, self.hidden_size, self.pad_idx)
        setattr(self, "rnn", dec)
        return dec

    # ---------------- forward / inference ----------------
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        images = images.to(self.device)
        feats = self.spatial_encoder(images)  # [B, C, H, W]
        B, C, H, W = feats.shape
        memory = feats.view(B, C, H * W).permute(0, 2, 1).contiguous()  # [B, S, C]

        if memory.size(1) > 100:
            memory = memory[:, :100, :]

        if captions is not None:
            captions = self._normalize_captions(captions.to(self.device).long())
            logits, new_hidden = self.decoder(memory, captions, hidden_state)
            return logits, new_hidden

        # one-step logits from <SOS> for compatibility
        inputs = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)
        logits, new_hidden = self.decoder(memory, inputs, hidden_state)
        return logits, new_hidden

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        images = images.to(self.device)
        feats = self.spatial_encoder(images)  # [B, C, H, W]
        B, C, H, W = feats.shape
        memory = feats.view(B, C, H * W).permute(0, 2, 1).contiguous()  # [B, S, C]
        if memory.size(1) > 100:
            memory = memory[:, :100, :]

        inputs = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)
        tokens = []
        hidden_state = None

        for _ in range(self.max_decode_len):
            logits, hidden_state = self.decoder(memory, inputs, hidden_state)
            step_logits = logits[:, -1, :]
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            tokens.append(next_tok)
            inputs = torch.cat([inputs, next_tok], dim=1)
            if (next_tok == self.eos_idx).all() or (next_tok == self.pad_idx).all():
                break

        if tokens:
            return torch.cat(tokens, dim=1)
        return torch.empty(B, 0, dtype=torch.long, device=self.device)

    def init_zero_hidden(self, batch: int, device: torch.device):
        if hasattr(self.decoder, "init_zero_hidden"):
            return self.decoder.init_zero_hidden(batch, device)
        h = torch.zeros(batch, self.hidden_size, device=device)
        c = torch.zeros(batch, self.hidden_size, device=device)
        return h, c
