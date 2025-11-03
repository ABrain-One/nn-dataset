import math
from typing import Optional, Tuple, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- public API -------------------------
def supported_hyperparameters():
    return {'lr', 'momentum'}


# ------------------------- helpers -------------------------
def _first_int_leaf(x):
    """Return the first integer leaf (handles nested out_shape)."""
    if isinstance(x, (list, tuple)) and len(x):
        return _first_int_leaf(x[0])
    return int(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    """ResNet BasicBlock + Squeeze-and-Excitation."""
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        se_reduction: int = 16,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

        # Squeeze-and-Excitation
        hidden = max(1, planes // se_reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, planes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # apply SE
        scale = self.se(out)
        out = out * scale

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class EncoderSEResNet(nn.Module):
    """A small SE-ResNet-like encoder that outputs a single feature vector per image."""
    def __init__(self, in_ch: int = 3, hidden_size: int = 768, base_width: int = 64):
        super().__init__()
        norm = nn.BatchNorm2d

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            norm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # layers: (inplanes -> planes, blocks, stride)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, hidden_size)

    def _make_layer(self, inplanes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers: List[nn.Module] = []
        layers.append(SEBasicBlock(inplanes, planes, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(SEBasicBlock(planes, planes, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)      # [B, 512]
        x = self.proj(x)                  # [B, H]
        return x


class PositionalEncoding(nn.Module):
    """Standard sine-cos positional encoding for Transformer decoder (batch_first)."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class Net(nn.Module):
    """
    Image Captioning model:
      - Encoder: SE-ResNet-ish -> feature vector (H)
      - Decoder: TransformerDecoder (batch_first=True) conditioned on image feature as memory (length=1)
    """
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # infer channels from in_shape
        self.in_channels = int(in_shape[1]) if isinstance(in_shape, (list, tuple)) and len(in_shape) > 1 else 3
        # vocab size from possibly nested out_shape
        self.vocab_size = _first_int_leaf(out_shape)

        # hyperparams
        self.hidden_size = int(prm.get('hidden_size', 768))
        self.num_heads = int(prm.get('num_heads', 8))
        if self.hidden_size % self.num_heads != 0:
            # fix heads if needed
            for h in (16, 12, 8, 4, 2):
                if self.hidden_size % h == 0:
                    self.num_heads = h
                    break
        self.num_layers = int(prm.get('num_layers', 2))
        self.dropout = float(prm.get('dropout', 0.1))
        self.pad_idx = int(prm.get('pad_idx', 0))
        self.sos_idx = int(prm.get('sos_idx', 1))
        self.eos_idx = int(prm.get('eos_idx', 2))
        self.max_len = int(prm.get('max_len', 30))

        # encoder
        self.encoder = EncoderSEResNet(self.in_channels, self.hidden_size)

        # decoder components
        self.token_embed = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_idx)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.num_layers)
        self.pos_enc = PositionalEncoding(self.hidden_size, max_len=max(512, self.max_len + 5))
        self.readout = nn.Linear(self.hidden_size, self.vocab_size)

        self.criteria: Tuple[nn.Module, ...] = ()
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    # ---------- API helpers ----------
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm['lr']),
            betas=(float(prm.get('momentum', 0.9)), 0.999),
        )

    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Transformer doesn't use (h, c) but keep API compatibility
        z = torch.zeros(1, batch, self.hidden_size, device=device)
        return z, z

    # ---------- core ----------
    def _subsequent_mask(self, T: int) -> torch.Tensor:
        # [T, T] with True masked (upper triangle)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=self.device), diagonal=1)
        return mask

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        # returns memory: [B, 1, H]
        feats = self.encoder(images)            # [B, H]
        return feats.unsqueeze(1)               # [B, 1, H]

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Teacher forcing:
          captions: [B,T] or [B,1,T]
          returns logits [B,T-1,V], None
        Inference (captions=None): greedy decode
          returns logits [B,Tgen,V], tokens [B,Tgen]
        """
        B = images.size(0)
        memory = self._encode(images.to(self.device))  # [B,1,H]

        if captions is not None:
            if captions.dim() == 3:
                captions = captions[:, 0, :]  # [B,T]

            # teacher forcing: inputs=tokens[:-1], targets=tokens[1:]
            inp = captions[:, :-1].to(self.device)     # [B,T-1]
            tgt = captions[:, 1:].to(self.device)      # [B,T-1]

            tgt_emb = self.token_embed(inp)            # [B,T-1,H]
            tgt_emb = self.pos_enc(tgt_emb)            # add PE
            tgt_mask = self._subsequent_mask(tgt_emb.size(1))  # [T-1,T-1]

            dec_out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)  # [B,T-1,H]
            logits = self.readout(dec_out)               # [B,T-1,V]
            return logits, None

        # Greedy generation
        tokens = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)  # [B,1]
        logits_steps = []
        for _ in range(self.max_len):
            tgt_emb = self.token_embed(tokens)          # [B,t,H]
            tgt_emb = self.pos_enc(tgt_emb)
            tgt_mask = self._subsequent_mask(tgt_emb.size(1))
            dec_out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)  # [B,t,H]
            step_logits = self.readout(dec_out[:, -1:, :])  # [B,1,V]
            logits_steps.append(step_logits)
            next_tok = step_logits.argmax(-1)           # [B,1]
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok.squeeze(1) == self.eos_idx).all():
                break

        logits = torch.cat(logits_steps, dim=1) if logits_steps else torch.zeros(B, 0, self.vocab_size, device=self.device)
        return logits, tokens[:, 1:]  # drop initial SOS

    # ---------- training loop ----------
    def learn(self, train_data):
        """
        train_data: iterable of (images, captions)
          images:  [B,C,H,W]
          captions:[B,T] or [B,1,T] (must include SOS/EOS; PAD = self.pad_idx)
        """
        assert self.criteria and self.optimizer is not None, "Call train_setup(prm) before learn()."
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            if captions.dim() == 3:
                captions = captions[:, 0, :]

            # teacher forcing split
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            logits, _ = self.forward(images, captions=inputs)  # [B,T-1,V]
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
