import math
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, Iterable, Optional, Tuple

def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device, *_, **__) -> None:
        super().__init__()
        self.device = device
        self.prm = prm or {}
        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._infer_vocab_size(out_shape)

        # hyperparams
        self.embed_dim = int(self.prm.get('embed_dim', 768))
        self.hidden_size = int(self.prm.get('hidden_size', 768))
        self.num_layers = int(self.prm.get('num_layers', 1))
        self.num_heads = int(self.prm.get('num_heads', 8))
        self.max_len = int(self.prm.get('max_length', 20))
        self.sos = int(self.prm.get('sos', 1))
        self.eos = int(self.prm.get('eos', 0))
        self.grad_clip = float(self.prm.get('grad_clip', 3.0))

        # --- Encoder: ViT-ish (patchify with stride 16, then TransformerEncoder) ---
        self.stem = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=16, stride=16)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=self.num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)

        # --- Decoder: token embedding + LSTM + projection ---
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size, self.num_layers, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

        # map pooled encoder features to initial LSTM state
        self.enc_to_h = nn.Linear(self.embed_dim, self.hidden_size)
        self.enc_to_c = nn.Linear(self.embed_dim, self.hidden_size)

        # training objects (lazy init in train_setup)
        self.criterion: Optional[nn.Module] = None
        self.criteria = None
        self.optimizer: Optional[optim.Optimizer] = None

        self.to(self.device)

    # ---------------- training plumbing ----------------
    def train_setup(self, prm: Dict[str, Any]):
        lr = float(prm.get('lr', 1e-3))
        beta1 = float(prm.get('momentum', 0.9))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.criteria = (self.criterion,)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data: Iterable[Tuple[torch.Tensor, torch.Tensor]] | Dict[str, torch.Tensor], *_, **__):
        # Accepts either a dict {"images","captions"} or an iterable of (images, captions)
        if self.optimizer is None or self.criterion is None:
            self.train_setup(self.prm)

        self.train()

        def _step(images: torch.Tensor, captions: torch.Tensor):
            images = images.to(self.device).float()
            captions = captions.to(self.device).long()
            if captions.size(1) <= 1:
                return  # nothing to predict

            # teacher forcing: predict next token
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            logits, _ = self.forward(images, inputs)  # [B, T-1, V]
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            self.optimizer.step()

        if isinstance(train_data, dict):
            _step(train_data['images'], train_data['captions'])
        else:
            for images, captions in train_data:
                _step(images, captions)

    # ---------------- forward / inference ----------------
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        # Encode images to sequence of patch embeddings
        seq, pooled = self.encode(images)                      # seq: [B, N, E], pooled: [B, E]

        # init LSTM state from image features if not provided
        if hidden_state is None:
            h0 = torch.tanh(self.enc_to_h(pooled)).unsqueeze(0).repeat(self.num_layers, 1, 1)  # [L,B,H]
            c0 = torch.tanh(self.enc_to_c(pooled)).unsqueeze(0).repeat(self.num_layers, 1, 1)
            hidden_state = (h0, c0)

        if captions is not None:
            # teacher-forcing decode
            emb = self.embedding(captions.to(self.device).long())  # [B,T,E]
            out, hidden_state = self.lstm(emb, hidden_state)       # [B,T,H]
            logits = self.fc_out(out)                               # [B,T,V]
            return logits, hidden_state

        # Greedy decode
        B = images.size(0)
        tokens = torch.full((B, 1), self.sos, dtype=torch.long, device=self.device)
        h = hidden_state
        for _ in range(self.max_len):
            emb = self.embedding(tokens[:, -1:])                   # last token -> [B,1,E]
            out, h = self.lstm(emb, h)                             # [B,1,H]
            step_logits = self.fc_out(out[:, -1])                  # [B,V]
            next_tok = step_logits.argmax(dim=-1, keepdim=True)    # [B,1]
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == self.eos).all():
                break
        return tokens, h

    # ---------------- encoder + positional encoding ----------------
    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(images.to(self.device).float())              # [B,E,H',W']
        B, E, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)                           # [B, N, E], N=H'*W'

        # sinusoidal positional encoding with correct N
        pos = self._sinusoidal_positional_encoding(x.size(1), self.embed_dim, x.device)  # [1,N,E]
        x = x + pos

        x = self.transformer(x)                                    # [B, N, E]
        pooled = x.mean(dim=1)                                     # [B, E]
        return x, pooled

    @staticmethod
    def _sinusoidal_positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
        pe = torch.zeros(length, dim, device=device)
        position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, length, dim]

    # ---------------- compatibility helpers ----------------
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        return h0, c0

    @staticmethod
    def _infer_in_channels(in_shape) -> int:
        if isinstance(in_shape, (tuple, list)):
            # common: (B, C, H, W)
            if len(in_shape) >= 4:
                return int(in_shape[1])
            # or (C, H, W)
            if len(in_shape) == 3:
                return int(in_shape[0])
        return 3

    @staticmethod
    def _infer_vocab_size(out_shape) -> int:
        if isinstance(out_shape, int):
            return out_shape
        if isinstance(out_shape, (tuple, list)):
            # many pipelines use out_shape[0][0]
            try:
                return int(out_shape[0][0])
            except Exception:
                try:
                    return int(out_shape[0])
                except Exception:
                    pass
        return int(out_shape)
