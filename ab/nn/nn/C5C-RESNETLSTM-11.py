import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple


def supported_hyperparameters():
    return {"lr", "momentum"}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: Dict[str, Any], device: torch.device):
        super().__init__()
        self.device = device
        prm = prm or {}

        # ---- basic config ----
        self.grad_clip_value = float(prm.get("grad_clip", 3.0))
        self.dropout_p = float(prm.get("dropout", 0.1))
        self.hidden_size = int(prm.get("hidden_size", 768))
        self.num_layers = int(prm.get("num_layers", 1))
        self.decoder_type = prm.get("decoder_type", "lstm")
        self.sos_idx = int(prm.get("sos_idx", 1))
        self.pad_idx = int(prm.get("pad_idx", 0))
        self.max_decode_len = int(prm.get("max_decode_len", 50))

        # ---- in/out shapes ----
        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)

        # ---- backbone (kept simple and torchvision-free) ----
        backbone_type = str(prm.get("backbone", "resnet")).lower()
        if backbone_type == "mobilenetv3":
            # simplified "fire"-style blocks
            self.spatial_encoder = nn.Sequential(
                self._make_fire_layer(self.in_channels, 64, float(prm.get("width_mult", 1.0))),
                self._make_fire_layer(64, 128, float(prm.get("width_mult", 1.0))),
            )
            enc_out_channels = 128
        elif backbone_type == "efficientnet":
            # tiny conv stack; no partial() misuse
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
        else:  # "resnet"-like stem
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

        self.feature_dim = enc_out_channels  # channels that come out of the encoder

        # ---- decoder ----
        self.decoder = self.build_decoder(self.hidden_size, self.num_layers)
        self.to(self.device)

    # ---------------- API helpers ----------------
    @staticmethod
    def _infer_in_channels(in_shape: Any) -> int:
        if isinstance(in_shape, (tuple, list)) and len(in_shape) >= 2 and isinstance(in_shape[1], int):
            return in_shape[1]
        return 3

    @staticmethod
    def _first_int(x: Any) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            # find the first integer nested anywhere
            for item in x:
                try:
                    return Net._first_int(item)
                except Exception:
                    continue
        return int(x)

    # ---------------- training setup ----------------
    def train_setup(self, prm):
        prm = prm or {}
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device)
        self.criteria = (self.criterion,)  # some harnesses expect this tuple
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data: Any):
        """Train the model on provided data (teacher forcing)."""
        self.train()
        if not hasattr(self, "optimizer"):
            self.train_setup(getattr(train_data, "prm", None))

        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device).long()

            # If captions come as [B, 1, T], squeeze the singleton dim
            if captions.dim() == 3 and captions.size(1) == 1:
                captions = captions[:, 0, :]

            # Need at least 2 tokens to create input/target shift
            if captions.dim() != 2 or captions.size(1) <= 1:
                continue

            # Shift for teacher forcing
            inputs = captions[:, :-1]   # feed to decoder
            targets = captions[:, 1:]   # predict next token

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
        """Very small 'fire' style block; channel math uses ints to avoid type errors."""
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
            # Minimal, working Transformer decoder (batch_first=True)
            class CaptionTransformerDecoder(nn.Module):
                def __init__(self, d_model: int, num_heads: int, vocab_size: int, pad_idx: int, dropout: float):
                    super().__init__()
                    self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
                    layer = nn.TransformerDecoderLayer(
                        d_model=d_model, nhead=num_heads, dim_feedforward=min(3072, d_model * 4),
                        dropout=dropout, batch_first=True
                    )
                    self.dec = nn.TransformerDecoder(layer, num_layers=2)
                    self.proj = nn.Linear(d_model, vocab_size)

                def _causal_mask(self, T: int, device: torch.device):
                    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), 1)

                def forward(self, features: torch.Tensor, inputs: torch.Tensor, hidden_state: Optional[Tuple] = None):
                    # features: [B, S, C], inputs: [B, T]
                    emb = self.embed(inputs)  # [B, T, C]
                    tgt_mask = self._causal_mask(emb.size(1), emb.device)
                    out = self.dec(tgt=emb, memory=features, tgt_mask=tgt_mask)  # [B, T, C]
                    logits = self.proj(out)  # [B, T, V]
                    return logits, None

            dec = CaptionTransformerDecoder(
                d_model=self.hidden_size, num_heads=8, vocab_size=self.vocab_size,
                pad_idx=self.pad_idx, dropout=self.dropout_p
            )
            setattr(self, "rnn", dec)
            return dec

        # Default: compact LSTM decoder
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
                features: torch.Tensor,           # [B, S, C]   (not used in this minimal decoder)
                inputs: torch.Tensor,             # [B, T]
                hidden_state: Optional[Tuple] = None,
            ) -> Tuple[torch.Tensor, Tuple]:
                B, T = inputs.size(0), inputs.size(1)
                h, c = (self.init_zero_hidden(B, inputs.device) if hidden_state is None else hidden_state)

                outputs = []
                for t in range(T):
                    token_t = inputs[:, t]
                    emb_t = self.embed(token_t)           # [B, H]
                    h, c = self.lstm_cell(emb_t, (h, c))  # each step
                    logits_t = self.proj_to_vocab(h)      # [B, V]
                    outputs.append(logits_t)

                logits = torch.stack(outputs, dim=1)      # [B, T, V]
                return logits, (h, c)

        dec = LSTMDecoder(self.vocab_size, self.hidden_size, self.pad_idx)
        setattr(self, "rnn", dec)
        return dec

    # ---------------- forward ----------------
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Train (teacher forcing): captions is [B, T_in], returns logits [B, T_in, V].
        Inference (greedy): captions is None, returns generated ids [B, T_out] (no <SOS>).
        """
        # Encode image to memory tokens
        feats = self.spatial_encoder(images.to(self.device))  # [B, C, H, W]
        B, C, H, W = feats.shape
        memory = feats.view(B, C, H * W).permute(0, 2, 1).contiguous()  # [B, S, C]

        # For stability in small decoders, cap sequence length
        if memory.size(1) > 100:
            memory = memory[:, :100, :]

        # Training / teacher-forcing path
        if captions is not None:
            assert captions.dim() == 2, "captions should be [B, T]"
            logits, new_hidden = self.decoder(memory, captions.to(self.device), hidden_state)
            return logits, new_hidden

        # Greedy decoding path
        inputs = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)
        generated = []

        for _ in range(self.max_decode_len):
            logits, hidden_state = self.decoder(memory, inputs, hidden_state)  # [B, t, V]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)         # [B, 1]
            generated.append(next_token)
            inputs = torch.cat([inputs, next_token], dim=1)
            if (next_token == self.pad_idx).all():  # stop if all hit PAD/EOS index
                break

        gen = torch.cat(generated, dim=1) if generated else torch.empty(B, 0, dtype=torch.long, device=self.device)
        return gen, hidden_state
