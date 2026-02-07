import torch
import torch.nn as nn
import numpy as np
import gc
import torch.utils.checkpoint as cp
from torch.nn import MaxPool2d

# -------------------------------------------------
# AMP helpers
# -------------------------------------------------
from torch.amp import autocast, GradScaler
def autocast_ctx(enabled=True):
    return autocast("cuda", enabled=enabled)
def make_scaler(enabled=True):
    return GradScaler("cuda", enabled=enabled)
def supported_hyperparameters():
    return {"lr", "dropout", "momentum"}

# -------------------------------------------------
# Main Conv Block
# -------------------------------------------------
def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0, element_list=['Conv2d', 'MaxPool2d', 'BatchNorm2d', 'ReLU', 'Dropout2d']):
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
    )
# -------------------------------------------------
# Fractal Block
# -------------------------------------------------
class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.num_columns = int(num_columns)
        self.loc_drop_prob = float(loc_drop_prob)
        depth = 2 ** max(self.num_columns - 1, 0)
        blocks = []
        for i in range(depth):
            level = nn.ModuleList()
            for j in range(self.num_columns):
                if (i + 1) % (2 ** j) == 0:
                    in_ch_ij = in_channels if (i + 1 == 2 ** j) else out_channels
                    # -------- SKIP CONNECTION BLOCK --------
                    level.append(
                        nn.Sequential(
                             nn.Conv2d(in_ch_ij, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                       nn.ReLU(inplace=True)
                    )
                    )
                else:
                    level.append(nn.Identity())
            blocks.append(level)
        self.blocks = nn.ModuleList(blocks)
        self.use_checkpoint_per_subblock = False

    def forward(self, x):
        outs = [x for _ in range(self.num_columns)]
        for level in self.blocks:
            new_outs = outs.copy()
            active_cols = []
            active_outs = []
            for col_idx, blk in enumerate(level):
                inp = outs[col_idx]
                if self.use_checkpoint_per_subblock:
                    out = cp.checkpoint(blk, inp, use_reentrant=False)
                else:
                    out = blk(inp)
                new_outs[col_idx] = out
                if not isinstance(blk, nn.Identity):
                    active_cols.append(col_idx)
                    active_outs.append(out)

            # ---- FRACTAL MERGE ----
            if len(active_outs) > 1:
                joined = torch.stack(active_outs, dim=0).mean(dim=0)
                for col_idx in active_cols:
                    new_outs[col_idx] = joined
            outs = new_outs
        return outs[0]
# -------------------------------------------------
# Fractal Unit
# -------------------------------------------------
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
# Network
# -------------------------------------------------
class Net(nn.Module):
    param_count_threshold: int = 80_000_000

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.glob_drop_ratio = 0.5
        self.loc_drop_prob = 0.15
        self.use_amp = False
        self.use_checkpoint = False

        self.param_count_threshold = int(prm.get("param_count_threshold", self.param_count_threshold))

        # -------- FILLED BY GENERATOR --------
        N = 3
        num_columns = 1
        element_list = ['Conv2d', 'MaxPool2d', 'BatchNorm2d', 'ReLU', 'Dropout2d']

        dropout_prob = float(prm["dropout"])
        self.fractal_fn(
            N=N,
            num_columns=num_columns,
            dropout_prob=dropout_prob,
            in_shape=in_shape,
            out_shape=out_shape,
            device=device,
            element_list=element_list,
        )

        # Adapt precision & checkpointing
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if param_count > self.param_count_threshold:
            self.use_amp = True
            self.use_checkpoint = True

        for m in self.modules():
            if isinstance(m, FractalUnit):
                m.use_checkpoint_whole = self.use_checkpoint
            if isinstance(m, FractalBlock):
                m.use_checkpoint_per_subblock = self.use_checkpoint

        self._scaler = make_scaler(enabled=self.use_amp)

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    @staticmethod
    def _parse_in_shape(in_shape):
        if len(in_shape) == 4:
            _, C, H, W = in_shape
        elif len(in_shape) == 3:
            C, H, W = in_shape
        else:
            raise ValueError(f"Invalid in_shape: {in_shape}")
        return int(C), int(H), int(W)

    @staticmethod
    def _norm4d(x):
        if x.dim() == 4:
            return x
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        raise ValueError(f"Invalid input shape {x.shape}")

    # -------------------------------------------------
    # Builder
    # -------------------------------------------------
    def fractal_fn(self, N, num_columns, dropout_prob, in_shape, out_shape, device, element_list):
        self.N = int(N)
        self.num_columns = int(num_columns)
        self.dropout_prob = float(dropout_prob)
        self.element_list = element_list

        C, H, W = self._parse_in_shape(in_shape)

        channels = [64 * (2 ** (i if i != 4 else i - 1)) for i in range(self.N)]
        dropout_probs = [min(0.5, dropout_prob + 0.1 * i) for i in range(self.N)]

        self.features = nn.Sequential()
        in_channels = C

        for i, out_channels in enumerate(channels):
            unit = FractalUnit(
                in_channels,
                out_channels,
                self.num_columns,
                self.loc_drop_prob,
                dropout_probs[i],
            )
            self.features.add_module(f"unit{i + 1}", unit)
            in_channels = out_channels

        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            feats = self.features(dummy)
            in_features = feats.view(1, -1).shape[1]

        self.output = nn.Linear(in_features, int(out_shape[0]))
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self, x):
        if next(self.parameters()).device != x.device:
            self.to(x.device)

        x = x.to(torch.float32)
        x = self._norm4d(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
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