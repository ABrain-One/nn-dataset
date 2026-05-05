import torch
import torch.nn as nn
import numpy as np
import os
import gc
import traceback
import torchvision
from torch.nn import MaxPool2d
import torch.utils.checkpoint as cp

class TorchVision(nn.Module):

    def __init__(self, model: str, weights: str='DEFAULT', unwrap: bool=True, truncate: int=2, split: bool=False):
        import torchvision
        super().__init__()
        if hasattr(torchvision.models, 'get_model'):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.split:
            y = [x]
            y.extend((m(y[-1]) for m in self.m))
            return y
        return self.m(x)

def adaptive_pool_flatten(x):
    if x.ndim == 4:
        return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    elif x.ndim == 3:
        return x.mean(dim=1)
    return x
from torch.amp import autocast, GradScaler

def autocast_ctx(enabled=True):
    return autocast('cuda', enabled=enabled)

def make_scaler(enabled=True):
    return GradScaler('cuda', enabled=enabled)

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias), nn.SiLU(inplace=True), nn.BatchNorm2d(out_channels))

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
                if (i + 1) % 2 ** j == 0:
                    in_ch_ij = in_channels if i + 1 == 2 ** j else out_channels
                    level.add_module(f'subblock{j + 1}', nn.Conv2d(in_ch_ij, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
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

class Net(nn.Module):
    param_count_threshold: int = 80000000

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.dropout_prob = float(prm['dropout'])
        self.use_amp = False
        self.use_checkpoint = False
        self.param_count_threshold = int(prm.get('param_count_threshold', self.param_count_threshold))
        self.backbone1 = TorchVision('efficientnet_b3', 'DEFAULT', True, 1).to(device)
        self.backbone2 = TorchVision('efficientnet_b0', 'DEFAULT', True, 1).to(device)
        N = 3
        num_columns = 3
        self.build_fractal_stream(N, num_columns, in_shape)
        self.to(self.device)
        self.infer_dimensions(in_shape, out_shape[0])
        param_count = sum((p.numel() for p in self.parameters() if p.requires_grad))
        if param_count > self.param_count_threshold:
            self.use_amp = True
            self.use_checkpoint = True
        for m in self.modules():
            if isinstance(m, FractalUnit):
                m.use_checkpoint_whole = self.use_checkpoint
            if isinstance(m, FractalBlock):
                m.use_checkpoint_per_subblock = self.use_checkpoint
        self._scaler = make_scaler(enabled=self.use_amp)

    def build_fractal_stream(self, N, num_columns, in_shape):
        C, H, W = self._parse_in_shape(in_shape)
        channels = [64 * 2 ** i for i in range(N)]
        drop_probs = [min(0.5, self.dropout_prob + 0.1 * i) for i in range(N)]
        self.features = nn.Sequential()
        in_channels = C
        for i, out_channels in enumerate(channels):
            unit = FractalUnit(in_channels=in_channels, out_channels=out_channels, num_columns=num_columns, loc_drop_prob=0.15, dropout_prob=drop_probs[i])
            self.features.add_module(f'unit{i + 1}', unit)
            in_channels = out_channels

    def infer_dimensions(self, in_shape, num_classes):
        C, H, W = self._parse_in_shape(in_shape)
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W, device=self.device).to(self.device)
            x = dummy.clone().detach()
            x = self.features(dummy)
            base_dim = x.view(1, -1).size(1)
            x1 = self.backbone1(dummy)
            x1 = adaptive_pool_flatten(x1)
            backbone1_dim = x1.size(1)
            x2 = self.backbone2(dummy)
            x2 = adaptive_pool_flatten(x2)
            backbone2_dim = x2.size(1)
        total_dim = base_dim + backbone1_dim + backbone2_dim
        self.fc = nn.Linear(total_dim, num_classes)

    def _parse_in_shape(self, in_shape):
        if len(in_shape) == 4:
            _, C, H, W = in_shape
        elif len(in_shape) == 3:
            C, H, W = in_shape
        else:
            raise ValueError('Input shape must be (C, H, W) or (N, C, H, W)')
        return (C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if next(self.parameters()).device != x.device:
            self.to(x.device)
        x = x.to(torch.float32)
        x = x.permute(0, 2, 3, 1) if x.dim() == 5 else x
        x = x.contiguous()
        x_base = self.features(x)
        x_base = x_base.view(x_base.size(0), -1)
        x1 = self.backbone1(x)
        x1 = adaptive_pool_flatten(x1)
        x2 = self.backbone2(x)
        x2 = adaptive_pool_flatten(x2)
        x = torch.cat([x_base, x1, x2], dim=1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.prm = prm
        self._batch_size = int(prm['batch'])
        self._epochs = int(prm['epoch'])
        self._transform = prm['transform']

        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'], weight_decay=0.0001)
        self._scaler = make_scaler(enabled=self.use_amp)

    def learn(self, train_data):
        self.train()
        scaler = self._scaler
        train_iter = iter(train_data)
        try:
            for batch_idx, (inputs, targets) in enumerate(train_iter):
                inputs, targets = (inputs.to(self.device), targets.to(self.device))
                self.optimizer.zero_grad(set_to_none=True)
                with autocast_ctx(enabled=self.use_amp):
                    outputs = self(inputs)
                    loss = self.criterion(outputs, targets)
                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
        finally:
            del train_data
            gc.collect()