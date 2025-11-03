import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- public API -------------------------
def supported_hyperparameters():
    return {'lr', 'momentum'}


# ------------------------- utilities -------------------------
def _deep_first_int(x: Union[int, Tuple, List]) -> int:
    """Pull the first integer-like leaf out of a possibly nested out_shape."""
    if isinstance(x, (tuple, list)) and len(x) > 0:
        return _deep_first_int(x[0])
    return int(x)


# ------------------------- EfficientNet-ish encoder blocks -------------------------
def _make_divisible(v: int, divisor: int = 8) -> int:
    return int(math.ceil(v / divisor) * divisor)


class SqueezeExcite(nn.Module):
    def __init__(self, in_ch: int, squeeze: int = 4):
        super().__init__()
        hidden = max(1, in_ch // squeeze)
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, in_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = self.fc(w)
        return x * w


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1, act=nn.SiLU):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            act(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class StochasticDepth(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x + residual
        keep = 1.0 - self.p
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x + residual * mask


@dataclass
class MBConvConfig:
    expand_ratio: int
    kernel: int
    stride: int
    in_ch: int
    out_ch: int
    num_layers: int
    se_ratio: float = 0.25


@dataclass
class FusedMBConvConfig:
    expand_ratio: int
    kernel: int
    stride: int
    in_ch: int
    out_ch: int
    num_layers: int
    se_ratio: float = 0.0  # usually fused doesn't use SE


class MBConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int, se_ratio: float, sd_prob: float):
        super().__init__()
        hidden = int(in_ch * expand_ratio)
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.sd = StochasticDepth(sd_prob) if self.use_res and sd_prob > 0 else None

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, hidden, k=1, s=1))
        # depthwise
        layers.append(ConvBNAct(hidden, hidden, k=3, s=stride, groups=hidden))
        # SE
        if se_ratio and se_ratio > 0:
            layers.append(SqueezeExcite(hidden, squeeze=int(1 / se_ratio)))
        # project
        layers.append(nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        ))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            if self.sd is None:
                out = out + x
            else:
                out = self.sd(out, x)
        return out


class FusedMBConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int, se_ratio: float, sd_prob: float):
        super().__init__()
        hidden = int(in_ch * expand_ratio)
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.sd = StochasticDepth(sd_prob) if self.use_res and sd_prob > 0 else None

        layers = []
        # fused conv does expansion by regular conv
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, hidden, k=3, s=stride))
            layers.append(nn.Sequential(
                nn.Conv2d(hidden, out_ch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_ch),
            ))
        else:
            layers.append(ConvBNAct(in_ch, out_ch, k=3, s=stride))

        # (optional) SE — often not used in fused, but support if se_ratio>0
        if se_ratio and se_ratio > 0:
            layers.append(SqueezeExcite(out_ch, squeeze=int(1 / se_ratio)))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            if self.sd is None:
                out = out + x
            else:
                out = self.sd(out, x)
        return out


# ------------------------- Net: Encoder (EfficientNet-ish) + GRU Decoder -------------------------
class Net(nn.Module):
    """
    Encoder: EfficientNet-lite built from inverted residual settings.
    Decoder: GRU conditioned on image features (concatenated to each token embedding).
    """

    def __init__(self, in_shape, out_shape, prm: dict, device: torch.device, *_, **__):
        super().__init__()
        self.device = device

        # --- resolve shapes ---
        if isinstance(in_shape, (tuple, list)) and len(in_shape) >= 2:
            self.in_channels = int(in_shape[1])
        else:
            self.in_channels = 3

        self.vocab_size = _deep_first_int(out_shape)
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # --- hyperparams ---
        self.hidden_size = int(prm.get('hidden_size', 768))
        self.dropout = float(prm.get('dropout', 0.3))
        self.num_layers = int(prm.get('num_layers', 1))
        self.sd_prob = float(prm.get('stochastic_depth', 0.0))
        self.pad_idx = int(prm.get('pad_idx', 0))
        self.sos_idx = int(prm.get('sos_idx', 1))
        self.eos_idx = int(prm.get('eos_idx', 2))
        self.max_len = int(prm.get('max_len', 30))

        # --- default inverted residual setting (similar to your A77 gist) ---
        # FusedMBConvConfig(expand, k, s, in_ch, out_ch, layers)
        inv: List[Union[MBConvConfig, FusedMBConvConfig]] = prm.get("inverted_residual_setting", [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ])

        # --- stem ---
        stem_out = 24
        self.stem = ConvBNAct(self.in_channels, stem_out, k=3, s=2)

        # --- stages ---
        stages: List[nn.Module] = []
        total_layers = sum(cfg.num_layers for cfg in inv)
        layer_id = 0
        in_ch_running = stem_out
        for cfg in inv:
            stage_layers: List[nn.Module] = []
            for i in range(cfg.num_layers):
                stride = cfg.stride if i == 0 else 1
                sd_here = self.sd_prob * float(layer_id) / max(1, total_layers - 1)
                if isinstance(cfg, FusedMBConvConfig):
                    stage_layers.append(FusedMBConv(
                        in_ch=in_ch_running,
                        out_ch=cfg.out_ch,
                        stride=stride,
                        expand_ratio=cfg.expand_ratio,
                        se_ratio=cfg.se_ratio,
                        sd_prob=sd_here
                    ))
                else:
                    stage_layers.append(MBConv(
                        in_ch=in_ch_running,
                        out_ch=cfg.out_ch,
                        stride=stride,
                        expand_ratio=cfg.expand_ratio,
                        se_ratio=cfg.se_ratio,
                        sd_prob=sd_here
                    ))
                in_ch_running = cfg.out_ch
                layer_id += 1
            stages.append(nn.Sequential(*stage_layers))

        self.stages = nn.Sequential(*stages)

        # --- head to hidden_size ---
        self.head = nn.Sequential(
            nn.Conv2d(in_ch_running, _make_divisible(in_ch_running * 4), 1, bias=False),
            nn.BatchNorm2d(_make_divisible(in_ch_running * 4)),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(_make_divisible(in_ch_running * 4), self.hidden_size),
        )

        # --- decoder ---
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_idx)
        self.gru = nn.GRU(
            input_size=self.hidden_size * 2,   # token embed + image feature
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

        self.criteria: Tuple[nn.Module, ...] = ()
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    # ---------- training API ----------
    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm['lr']),
            betas=(float(prm.get('momentum', 0.9)), 0.999),
        )

    def learn(self, train_data: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Expects iterable of (images, captions).
        captions: [B,T] or [B,1,T]; we use teacher forcing (inputs=tokens[:,:-1], targets=tokens[:,1:]).
        """
        assert self.criteria and self.optimizer is not None, "Call train_setup(prm) before learn()."
        self.train()

        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            if captions.dim() == 3:
                captions = captions[:, 0, :]

            # teacher forcing split
            inputs = captions[:, :-1]   # [B, T-1]
            targets = captions[:, 1:]   # [B, T-1]

            logits, _ = self.forward(images, captions=inputs)  # [B, T-1, V]
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
            # you can yield/log loss here if desired

    # ---------- inference helpers ----------
    def init_zero_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        x = self.stages(x)
        x = self.head(x)  # [B, H]
        return x

    # ---------- forward ----------
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        If captions is provided (teacher forcing):
            Returns (logits [B, T, V], h_n [num_layers, B, H])
        Else (greedy decoding):
            Returns (logits [B, Tgen, V], tokens [B, Tgen])
        """
        B = images.size(0)
        img_feats = self._encode(images.to(self.device))                  # [B, H]

        if captions is not None:
            if captions.dim() == 3:
                captions = captions[:, 0, :]
            captions = captions.to(self.device)                            # [B, T]
            T = captions.size(1)
            if T == 0:
                empty_logits = torch.zeros(B, 0, self.vocab_size, device=self.device)
                return empty_logits, self.init_zero_hidden(B)

            emb = self.embedding(captions)                                 # [B, T, H]
            img_rep = img_feats.unsqueeze(1).expand(B, T, -1)              # [B, T, H]
            gru_in = torch.cat([emb, img_rep], dim=-1)                     # [B, T, 2H]

            h0 = hidden_state if hidden_state is not None else self.init_zero_hidden(B)
            out, h_n = self.gru(gru_in, h0)                                # out: [B, T, H]
            logits = self.fc(out)                                          # [B, T, V]
            return logits, h_n

        # Greedy generation
        T = int(self.max_len)
        tokens = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)
        h = hidden_state if hidden_state is not None else self.init_zero_hidden(B)
        logits_acc = []

        for _ in range(T):
            emb = self.embedding(tokens[:, -1:])                           # [B,1,H]
            img_rep = img_feats.unsqueeze(1)                               # [B,1,H]
            step_in = torch.cat([emb, img_rep], dim=-1)                    # [B,1,2H]
            out, h = self.gru(step_in, h)                                  # [B,1,H]
            step_logits = self.fc(out)                                     # [B,1,V]
            logits_acc.append(step_logits)
            next_tok = step_logits.argmax(-1)                               # [B,1]
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok.squeeze(1) == self.eos_idx).all():
                break

        logits = torch.cat(logits_acc, dim=1) if logits_acc else torch.zeros(B, 0, self.vocab_size, device=self.device)
        return logits, tokens[:, 1:]
