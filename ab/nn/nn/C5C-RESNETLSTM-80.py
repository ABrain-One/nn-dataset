import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Iterable, Optional, Tuple, Union


# ----------------------------
# Public API: supported hparams
# ----------------------------
def supported_hyperparameters():
    return {'lr', 'momentum'}


# ----------------------------
# Small utilities
# ----------------------------
def _infer_in_channels(in_shape: Union[Tuple[int, ...], Iterable[int]]) -> int:
    """
    Works with shapes like (C,H,W) or (B,C,H,W) and odd variants we've seen.
    We grab the 3rd-from-last dimension (channels) when possible.
    """
    if not isinstance(in_shape, (tuple, list)) or len(in_shape) == 0:
        return 3
    if len(in_shape) >= 3:
        return int(in_shape[-3])
    # Fallback: try first
    return int(in_shape[0])


def _infer_vocab_size(out_shape: Any) -> int:
    """
    Robustly find an int vocab size from deeply nested tuples/lists or a plain int.
    Examples seen in prior dumps:
      - out_shape = 5000
      - out_shape = (5000,)
      - out_shape = ((5000,),)
      - out_shape = [[ [ [5000] ] ]]
    """
    if isinstance(out_shape, int):
        return out_shape
    if isinstance(out_shape, (tuple, list)):
        for x in out_shape:
            v = _infer_vocab_size(x)
            if isinstance(v, int) and v > 0:
                return v
    raise ValueError(f"Could not infer vocab size from out_shape={out_shape!r}")


# ----------------------------
# Encoder
# ----------------------------
class EncoderCNN(nn.Module):
    """
    Lightweight CNN encoder that produces a single 768-d feature per image.
    """
    def __init__(self, in_channels: int, hidden_size: int = 768):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # downsample

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(512, hidden_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, C, H, W]
        x = self.features(images)            # [B, 512, 1, 1]
        x = x.flatten(1)                     # [B, 512]
        x = self.proj(x)                     # [B, hidden]
        return x


# ----------------------------
# Decoder
# ----------------------------
class DecoderLSTM(nn.Module):
    """
    Simple LSTM decoder with teacher forcing.
    """
    def __init__(self, vocab_size: int, hidden_size: int = 768, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0, c0

    def forward(
        self,
        enc_feats: torch.Tensor,                 # [B, hidden]
        captions: Optional[torch.Tensor] = None, # [B, T] (teacher forcing) or None (inference)
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_len: int = 30,
        sos_idx: int = 1,
        eos_idx: int = 2,
    ):
        """
        Returns:
            logits: [B, T, V] if captions provided, else [B, T_gen, V]
            hidden: (h, c)
            tokens: generated token ids (for inference), else None
        """
        B = enc_feats.size(0)
        device = enc_feats.device

        if hidden is None:
            hidden = self.init_hidden(B, device)

        if captions is not None:
            # Teacher forcing
            if captions.dim() == 3:      # some loaders provide shape [B, 1, T]
                captions = captions[:, 0, :]

            # Input to LSTM is word embeddings (shifted right)
            # x_in = captions[:, :-1]  -> targets = captions[:, 1:]
            x_in = captions[:, :-1].contiguous()
            targets = captions[:, 1:].contiguous()

            emb = self.embedding(x_in)   # [B, T-1, H]
            out, hidden = self.lstm(emb, hidden)  # [B, T-1, H]
            logits = self.fc(out)        # [B, T-1, V]
            return logits, hidden, None, targets
        else:
            # Greedy decoding
            inputs = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)  # [B, 1]
            tokens = []
            logits_steps = []

            for _ in range(max_len):
                emb = self.embedding(inputs[:, -1:])   # last token, shape [B, 1, H]
                out, hidden = self.lstm(emb, hidden)   # [B, 1, H]
                step_logits = self.fc(out)             # [B, 1, V]
                logits_steps.append(step_logits)

                next_token = step_logits.argmax(dim=-1)  # [B, 1]
                tokens.append(next_token)

                if (next_token == eos_idx).all():
                    break
                inputs = torch.cat([inputs, next_token], dim=1)

            if len(logits_steps) == 0:
                # edge case: immediately EOS
                logits = torch.zeros(B, 0, self.vocab_size, device=device)
                tokens_out = torch.zeros(B, 0, dtype=torch.long, device=device)
            else:
                logits = torch.cat(logits_steps, dim=1)  # [B, T_gen, V]
                tokens_out = torch.cat(tokens, dim=1)    # [B, T_gen]
            return logits, hidden, tokens_out, None


# ----------------------------
# Main model wrapper
# ----------------------------
class Net(nn.Module):
    """
    Executable, minimal image-captioning model with:
      - EncoderCNN -> 768-d feature
      - DecoderLSTM -> caption logits
    Preserves expected methods: train_setup, learn, forward, init_zero_hidden.
    """
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device

        # Hyperparams (with safe defaults)
        self.hidden_size = int(prm.get('hidden_size', 768))
        self.num_layers = int(prm.get('num_layers', 1))
        self.dropout = float(prm.get('dropout', 0.1))
        self.sos_index = int(prm.get('sos_index', 1))
        self.eos_index = int(prm.get('eos_index', 2))
        self.max_len = int(prm.get('max_len', 30))

        # Shapes
        in_channels = _infer_in_channels(in_shape)
        vocab_size = _infer_vocab_size(out_shape)

        # Public aliases used elsewhere in your stack
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_channels = in_channels
        self.vocab_size = vocab_size
        self.out_dim = vocab_size
        self.num_classes = vocab_size

        # Components
        self.encoder = EncoderCNN(in_channels=in_channels, hidden_size=self.hidden_size)
        self.decoder = DecoderLSTM(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

    # --- compatibility helpers ---
    def init_zero_hidden(self, batch_size, device=None):
        device = device or self.device
        return self.decoder.init_hidden(batch_size, device)

    # --- training wiring ---
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999)
        )

    def learn(self, train_data):
        """
        Expects an iterable of (images, captions) where:
          images: [B, C, H, W]
          captions: [B, T] or [B, 1, T]; pad token = 0; SOS/EOS handled externally or via prm.
        """
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)

            feats = self.encoder(images)  # [B, H]
            logits, _, _, targets = self.decoder(
                feats, captions=captions,
                max_len=self.max_len, sos_idx=self.sos_index, eos_idx=self.eos_index
            )
            # logits: [B, T-1, V], targets: [B, T-1]
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

            # Optionally yield loss for external logging
            yield loss.detach()

    # --- forward (train + inference) ---
    def forward(self, images, captions=None, hidden_state=None):
        """
        Train (teacher forcing): pass captions -> returns (logits, hidden_state)
        Inference (greedy): captions=None -> returns (logits, hidden_state)
        """
        images = images.to(self.device)
        feats = self.encoder(images)  # [B, H]

        if captions is not None:
            captions = captions.to(self.device)
        if hidden_state is not None:
            hidden_state = (hidden_state[0].to(self.device), hidden_state[1].to(self.device))

        logits, hidden, tokens, targets = self.decoder(
            feats, captions=captions, hidden=hidden_state,
            max_len=self.max_len, sos_idx=self.sos_index, eos_idx=self.eos_index
        )
        # For training, tokens=None; for inference, targets=None
        return logits, hidden
