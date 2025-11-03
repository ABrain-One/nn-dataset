import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Iterable


def supported_hyperparameters():
    return {'lr', 'momentum'}


def _flatten(xs: Union[int, Iterable]):
    if isinstance(xs, (tuple, list)):
        for x in xs:
            yield from _flatten(x)
    else:
        yield xs


def _infer_vocab_size(out_shape) -> int:
    # Robustly extract the first int from possibly nested shapes
    for v in _flatten(out_shape):
        try:
            return int(v)
        except Exception:
            continue
    return int(out_shape)  # last resort


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, hidden_size, 3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, hidden]
        x = self.net(x)
        return x.flatten(1)


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_zero_hidden(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch, self.hidden_size, device=device)

    def forward(
        self,
        captions: torch.Tensor,          # [B, T]
        hidden_state: Optional[torch.Tensor] = None,  # [1, B, H]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(captions)         # [B, T, H]
        out, h = self.gru(x, hidden_state)
        logits = self.fc(out)            # [B, T, V]
        return logits, h


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device
        self.prm = prm or {}

        # ---- API aliases (common in this codebase) ----
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3
        self.vocab_size = _infer_vocab_size(out_shape)
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # tokens / decoding defaults
        self.sos = int(self.prm.get('sos', 0))
        self.eos = int(self.prm.get('eos', 1))
        self.max_len = int(self.prm.get('max_len', 20))

        # modules
        hidden_size = int(self.prm.get('hidden_size', 512))
        self.encoder = CNNEncoder(self.in_channels, hidden_size=hidden_size)
        self.decoder = GRUDecoder(self.vocab_size, hidden_size=hidden_size)

        # expose rnn-style helper for compatibility
        self.init_zero_hidden = self.decoder.init_zero_hidden

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.criteria = (self.criterion,)  # some trainers expect a tuple
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get('lr', 1e-3)),
            betas=(float(prm.get('momentum', 0.9)), 0.999),
        )

    def learn(self, train_data):
        # Placeholder; training loops are dataset/loader specific.
        # Kept for API compatibility.
        pass

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None
    ):
        """
        Training:  images [B, C, H, W], captions [B, T] -> logits [B, T, V]
        Inference: images [B, C, H, W] -> generated [B, <=max_len+1]
        """
        images = images.to(self.device, non_blocking=True).float()

        # Encode image (not used directly by GRU here but keeps the interface stable)
        _ = self.encoder(images)  # can be extended into attention later

        if captions is not None:
            if captions.dim() == 3:  # e.g. [B, 1, T]
                captions = captions.squeeze(1)
            captions = captions.to(self.device).long()
            if hidden_state is not None:
                hidden_state = hidden_state.to(self.device)
            logits, hidden_state = self.decoder(captions, hidden_state)
            return logits, hidden_state

        # Greedy decoding (no attention): start from SOS and roll GRU
        B = images.size(0)
        seq = torch.full((B, 1), self.sos, device=self.device, dtype=torch.long)
        h = self.decoder.init_zero_hidden(B, self.device)

        for _ in range(self.max_len):
            logits, h = self.decoder(seq[:, -1:], h)    # step with last token
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)  # [B,1]
            seq = torch.cat([seq, next_tok], dim=1)
            if (next_tok == self.eos).all():
                break

        return seq, None
