import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision
from torch.nn import MaxPool2d
from torch.amp import autocast, GradScaler

class TorchVision(nn.Module):

    def __init__(self, model: str, weights: str='DEFAULT', unwrap: bool=True, truncate: int=1, in_channels: int=3):
        super().__init__()
        self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1) if in_channels != 3 else nn.Identity()
        if hasattr(torchvision.models, 'get_model'):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=booleans(weights))
        if unwrap:
            layers = []
            for name, module in self.m.named_children():
                if 'aux' in name.lower():
                    continue
                layers.append(module)
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
        else:
            self.m.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.m(self.adapter(x))

def adaptive_pool_flatten(x):
    return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
from torch.amp import autocast, GradScaler

def autocast_ctx(enabled=True):
    return autocast('cuda', enabled=enabled)

def make_scaler(enabled=True):
    return GradScaler('cuda', enabled=enabled)

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

class FractalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.num_columns = int(num_columns)
        depth = 2 ** max(self.num_columns - 1, 0)
        blocks = []
        for i in range(depth):
            level = nn.ModuleList()
            for j in range(self.num_columns):
                if (i + 1) % 2 ** j == 0:
                    in_ch_ij = in_channels if i + 1 == 2 ** j else out_channels
                    level.append(nn.Sequential(nn.Conv2d(in_ch_ij, out_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity(), nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()))
            blocks.append(level)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = [x] * self.num_columns
        for level_block in self.blocks:
            outs_i = [blk(inp) for blk, inp in zip(level_block, outs)]
            stacked = torch.stack(outs_i, dim=0).sum(dim=0)
            outs = [stacked] * len(level_block)
        return outs[0]

class FractalUnit(nn.Module):

    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.block = FractalBlock(in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob)
        self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.block(x))

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.use_amp = prm.get('use_amp', False)
        dropout_prob = float(prm.get('dropout', 0.1))
        self.backbone_a = TorchVision('shufflenet_v2_x0_5', in_channels=3, weights='DEFAULT').to(device)
        self.backbone_b = TorchVision('shufflenet_v2_x0_5', in_channels=3, weights='DEFAULT').to(device)
        curr_ch = 1024
        channels = [64]
        self.features = nn.Sequential()
        for i, out_ch in enumerate(channels):
            self.features.add_module(f'unit{i + 1}', FractalUnit(curr_ch, out_ch, 2, 0.15, dropout_prob))
            curr_ch = out_ch
        self.infer_dimensions_dynamically(in_shape, out_shape[0])
        self._scaler = make_scaler(enabled=self.use_amp)

    def infer_dimensions_dynamically(self, in_shape, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            C = in_shape[1] if len(in_shape) == 4 else in_shape[0]
            dummy = torch.zeros(1, C, 224, 224).to(self.device)
            x_a = self.backbone_a(dummy)
            x_b = self.backbone_b(dummy)
            f_bf = self.features(x_b)
            dim_fused = f_bf.size(1)
        self.classifier = nn.Linear(dim_fused + x_a.size(1), num_classes)
        self.train()

    def forward(self, x, is_probing=False):
        x_a = self.backbone_a(x)
        x_b = self.backbone_b(x)
        f_a = adaptive_pool_flatten(x_a)
        f_b = adaptive_pool_flatten(self.features(x_b))
        fused = torch.cat([f_a, f_b], dim=1)
        return fused if is_probing else self.classifier(fused)

    def train_setup(self, prm):
        self._batch_size = int(prm['batch'])
        self._epochs = int(prm['epoch'])
        self._transform = prm['transform']
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
        self._scaler = make_scaler(enabled=self.use_amp)

    def learn(self, train_data):
        self.train()
        scaler = self._scaler
        train_iter = iter(train_data)
        try:
            for batch_idx, (inputs, labels) in enumerate(train_iter):
                inputs, labels = (inputs.to(self.device), labels.to(self.device))
                self.optimizer.zero_grad(set_to_none=True)
                with autocast_ctx(enabled=self.use_amp):
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                    self.optimizer.step()
        finally:
            del train_data
            gc.collect()