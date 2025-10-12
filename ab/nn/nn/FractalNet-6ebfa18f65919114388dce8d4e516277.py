import torch
import torch.nn as nn
import numpy as np
import os
import gc
import traceback
from torch.nn import MaxPool2d
import torch.utils.checkpoint as cp

# -------------------------------------------------
# Handle PyTorch version differences for AMP
# -------------------------------------------------
from torch.amp import autocast, GradScaler
def autocast_ctx(enabled=True):
    return autocast("cuda", enabled=enabled)
def make_scaler(enabled=True):
    return GradScaler("cuda", enabled=enabled)

def supported_hyperparameters():
    return { 'lr', 'dropout', 'momentum' }


# -------------------------------------------------
# Conv block placeholder (with runner variables)
# -------------------------------------------------
def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0, element_list=['Conv2d', 'MaxPool2d', 'BatchNorm2d', 'ReLU', 'Dropout2d']):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )

# -------------------------------------------------
# Fractal Components
# -------------------------------------------------
class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.num_columns = int(num_columns)
        self.loc_drop_prob = float(loc_drop_prob)
        depth = 2 ** max(self.num_columns - 1, 0)
        blocks = []
        for i in range(depth):
            level = nn.Sequential()
            for j in range(self.num_columns):
                if (i + 1) % (2 ** j) == 0:
                    in_ch_ij = in_channels if (i + 1 == 2 ** j) else out_channels
                    level.add_module(
                        f"subblock{j + 1}",
                        nn.Conv2d(in_ch_ij, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
                    )
            blocks.append(level)
        self.blocks = nn.Sequential(*blocks)
        self.use_checkpoint_per_subblock = False

    def forward(self, x):
        outs = [x] * self.num_columns
        for level_block in self.blocks:
            if self.use_checkpoint_per_subblock:
                outs_i = [cp.checkpoint(blk, inp, use_reentrant=False) for blk, inp in zip(level_block, outs)]
            else:
                outs_i = [blk(inp) for blk, inp in zip(level_block, outs)]
            joined = torch.stack(outs_i, dim=0).mean(dim=0)
            outs[:len(level_block)] = [joined] * len(level_block)
        return outs[0]


class FractalUnit(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.block = FractalBlock(in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_checkpoint_whole = False

    def forward(self, x):
        if self.use_checkpoint_whole:
            x = cp.checkpoint(self.block, x, use_reentrant=False)
        else:
            x = self.block(x)
        return self.pool(x)


# -------------------------------------------------
# Net
# -------------------------------------------------
class Net(nn.Module):
    param_count_threshold: int = 80_000_000  # safeguard

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.glob_drop_ratio = 0.5
        self.loc_drop_prob = 0.15
        self.use_amp = False
        self.use_checkpoint = False

        self.param_count_threshold = int(prm.get("param_count_threshold", self.param_count_threshold))

        # filled by generator
        N = 3
        num_columns = 2
        element_list = ['Conv2d', 'MaxPool2d', 'BatchNorm2d', 'ReLU', 'Dropout2d']

        dropout_prob = float(prm['dropout'])
        self.fractal_fn(
            N=int(N),
            num_columns=int(num_columns),
            dropout_prob=dropout_prob,
            in_shape=in_shape,
            out_shape=out_shape,
            device=device,
            element_list=element_list
        )

        # adapt based on param count
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if param_count > self.param_count_threshold:
            self.use_amp = True
            self.use_checkpoint = True

        # propagate checkpoint flags
        for m in self.modules():
            if isinstance(m, FractalUnit):
                m.use_checkpoint_whole = self.use_checkpoint
            if isinstance(m, FractalBlock):
                m.use_checkpoint_per_subblock = self.use_checkpoint

        self._scaler = make_scaler(enabled=self.use_amp)

    # ---------- helpers ----------
    @staticmethod
    def _parse_in_shape(in_shape):
        if len(in_shape) == 4:
            _, C, H, W = in_shape
        elif len(in_shape) == 3:
            C, H, W = in_shape
        else:
            raise ValueError(f"in_shape must be length 3 or 4, got {in_shape}")
        return int(C), int(H), int(W)

    @staticmethod
    def _norm4d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        raise ValueError(f"Expected 4D/5D input to Conv2d, got {tuple(x.shape)}")

    # ---------- builder ----------
    def fractal_fn(self, N, num_columns, dropout_prob, in_shape: tuple, out_shape: tuple, device, element_list):
        self.N = int(N)
        self.num_columns = int(num_columns)
        self.dropout_prob = float(dropout_prob)
        self.element_list = element_list

        C, H, W = self._parse_in_shape(in_shape)

        channels = [64 * (2 ** (i if i != 4 else i - 1)) for i in range(self.N)]
        dropout_probs = [min(0.5, self.dropout_prob + 0.1 * i) for i in range(self.N)]

        self.features = nn.Sequential()
        in_channels = C
        for i, out_channels in enumerate(channels):
            unit = FractalUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                num_columns=self.num_columns,
                loc_drop_prob=self.loc_drop_prob,
                dropout_prob=dropout_probs[i]
            )
            self.features.add_module(f"unit{i + 1}", unit)
            in_channels = out_channels

        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W, dtype=torch.float32, device='cpu')
            feats = self.features.to('cpu')(dummy)
            in_features = feats.view(1, -1).shape[1]

        self.output = nn.Linear(in_features=in_features, out_features=int(out_shape[0]))
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------- forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if next(self.parameters()).device != x.device:
            self.to(x.device)
        x = x.to(torch.float32)
        x = self._norm4d(x)
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.output(x)

    # ---------- training setup ----------
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )
        self._scaler = make_scaler(enabled=self.use_amp)

    # ---------- learning loop ----------
    def learn(self, train_data):
        self.train()
        scaler = self._scaler
        train_iter = iter(train_data)
        try:
            for batch_idx, (inputs, labels) in enumerate(train_iter):
                    inputs = inputs.to(self.device).float()
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad(set_to_none=True)

                    with autocast_ctx(enabled=self.use_amp):
                        outputs = self(inputs)
                        loss = self.criterion(outputs, labels)

                    if not torch.isfinite(loss):
                        print(f"[WARN] Skipping batch {batch_idx} due to non-finite loss: {loss.item()}")
                        continue

                    if self.use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                        self.optimizer.step()
        finally:
            if hasattr(train_iter, 'shutdown'):
                train_iter.shutdown()
            del train_iter
            gc.collect()