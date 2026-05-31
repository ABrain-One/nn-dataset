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
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
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
    if x.ndim == 4:
        return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    if x.ndim == 3:
        return x.mean(dim=1)
    return x.flatten(1) if x.ndim > 2 else x
from torch.amp import autocast, GradScaler

def autocast_ctx(enabled=True):
    return autocast('cuda', enabled=enabled)

def make_scaler(enabled=True):
    return GradScaler('cuda', enabled=enabled)

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias), nn.BatchNorm2d(out_channels), nn.GELU(), nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias), nn.BatchNorm2d(out_channels), nn.GELU())

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
                    level.append(drop_conv3x3_block(in_ch_ij, out_channels, dropout_prob=dropout_prob))
            blocks.append(level)
        self.blocks = nn.ModuleList(blocks)
        self.use_checkpoint_per_subblock = False

    def forward(self, x):
        outs = [x] * self.num_columns
        for level_block in self.blocks:
            outs_i = [blk(inp) for blk, inp in zip(level_block, outs)]
            joined = torch.stack(outs_i, dim=0).mean(dim=0)
            outs[:len(level_block)] = [joined] * len(level_block)
        return outs[0]

class FractalUnit(nn.Module):

    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.block = FractalBlock(in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.block(x))

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.pattern = 'Ensemble_Backbones_to_Fractal'
        self.use_amp = prm.get('use_amp', False)
        self.dropout = nn.Dropout(prm.get('dropout', 0.5))
        self.backbone_a = TorchVision('efficientnet_b3', in_channels=3, weights='IMAGENET1K_V1').to(device)
        self.backbone_b = TorchVision('efficientnet_b3', in_channels=3, weights='IMAGENET1K_V1').to(device)
        self.features = nn.Sequential()
        curr_ch = 2816
        channels = [256, 512, 1024, 2048]
        for i, out_ch in enumerate(channels):
            self.features.add_module(f'unit{i + 1}', FractalUnit(curr_ch, out_ch, 4, 0.2, prm.get('momentum', 0.9)))
            curr_ch = out_ch
        self.infer_dimensions_dynamically(in_shape, out_shape[0])
        self.classifier = nn.Linear(self.feat_dim, out_shape[0])
        self._scaler = GradScaler(enabled=self.use_amp)

    def infer_dimensions_dynamically(self, in_shape, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            dummy = torch.randn(1, in_shape[1], in_shape[2], in_shape[3]).to(self.device)
            feat_a = adaptive_pool_flatten(self.backbone_a(dummy))
            feat_b = adaptive_pool_flatten(self.backbone_b(dummy))
            self.feat_dim = feat_a.size(1) + feat_b.size(1)
            self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.train()

    def forward(self, x: torch.Tensor, is_probing: bool=False):
        x = x.to(self.device)
        if x.dim() == 2:
            x = x[None, :, None, None].repeat(1, 3, 1, 1)
        elif x.dim() == 3:
            x = x[None, :, :, :].repeat(1, 3, 1, 1) if x.size(0) == 1 else x
        fa = self.backbone_a(x)
        fb = self.backbone_b(x)
        fa = adaptive_pool_flatten(fa)
        fb = adaptive_pool_flatten(fb)
        mid = self.dropout(torch.cat((fa, fb), dim=1))
        if is_probing:
            return mid
        return self.classifier(mid)

    def train_setup(self, prm):
        self.prm = prm
        self._batch_size = int(prm['batch'])
        self._epochs = int(prm['epoch'])
        self._transform = prm['transform']

        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9), weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def learn(self, data_loader):
        self.train()
        device = self.device
        scaler = self._scaler
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = (inputs.to(device), labels.to(device))
            self.optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(enabled=self.use_amp):
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
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
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item()}')
        self.scheduler.step()