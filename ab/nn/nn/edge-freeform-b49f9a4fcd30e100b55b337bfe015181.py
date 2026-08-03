import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision
from torch.nn import MaxPool2d
from torch.amp import autocast, GradScaler

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

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

def autocast_ctx(enabled=True):
    return autocast('cuda', enabled=enabled)

def make_scaler(enabled=True):
    return GradScaler('cuda', enabled=enabled)

class DSConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SEBlock(nn.Module):

    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.fc(x)
        return identity * x

class InvertedResidualBlock(nn.Module):

    def __init__(self, in_channels, expansion_factor, out_channels, stride):
        super(InvertedResidualBlock, self).__init__()
        expanded_dim = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels
        self.expand_conv = nn.Conv2d(in_channels, expanded_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_dim)
        self.dw_conv = nn.Conv2d(expanded_dim, expanded_dim, kernel_size=3, stride=stride, padding=1, groups=expanded_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(expanded_dim)
        self.pw_conv = nn.Conv2d(expanded_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x
        x = self.expand_conv(x)
        x = self.bn1(x)
        x = nn.functional.relu6(x)
        x = self.dw_conv(x)
        x = self.bn2(x)
        x = nn.functional.relu6(x)
        x = self.pw_conv(x)
        x = self.bn3(x)
        x = self.se(x)
        x += self.shortcut(identity)
        return x

class FractalUnit(nn.Module):

    def __init__(self, in_channels, out_channels, num_blocks, stride=1, expansion_factor=6):
        super(FractalUnit, self).__init__()
        layers = [InvertedResidualBlock(in_channels, expansion_factor, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(InvertedResidualBlock(out_channels, expansion_factor, out_channels, 1))
        self.layers = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.layers(x)
        return self.pool(x)

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.to(self.device)
        self.use_amp = prm.get('use_amp', False)
        dropout_prob = float(prm['dropout'])
        self.backbone_a = TorchVision('mobilenet_v3_small', in_channels=in_shape[1]).to(device)
        self.features = nn.Sequential(FractalUnit(in_shape[1], 64, num_blocks=2, stride=1, expansion_factor=6), FractalUnit(64, 128, num_blocks=2, stride=2, expansion_factor=6), DSConv(128, 128, stride=1), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.infer_dimensions_dynamically(in_shape, out_shape[0])
        self._scaler = make_scaler(enabled=self.use_amp)

    def infer_dimensions_dynamically(self, in_shape, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            C = in_shape[1] if len(in_shape) == 4 else in_shape[0]
            dummy = torch.zeros(1, C, 256, 256).to(self.device)
            output_feat = self.forward(dummy, is_probing=True)
            dim_fused = output_feat.shape[1]
        self.classifier = nn.Linear(dim_fused, num_classes)
        self.train()

    def forward(self, x: torch.Tensor, is_probing: bool=False) -> torch.Tensor:
        x = x.to(self.device)
        x_f = self.features(x)
        x_a = adaptive_pool_flatten(self.backbone_a(x))
        fused = torch.cat([x_f, x_a], dim=1)
        if is_probing:
            return fused
        return self.classifier(fused)

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
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with autocast_ctx(enabled=self.use_amp):
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                if not torch.isfinite(loss):
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