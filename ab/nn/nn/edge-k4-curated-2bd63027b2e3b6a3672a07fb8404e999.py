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

def autocast_ctx(enabled=True):
    return autocast('cuda', enabled=enabled)

def make_scaler(enabled=True):
    return GradScaler('cuda', enabled=enabled)

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

def conv_bn_act(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.conv = nn.Sequential(conv_bn_act(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, groups=in_channels), conv_bn_act(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        return self.conv(x)

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.use_amp = prm.get('use_amp', False)
        dropout_prob = float(prm.get('dropout', 0.2))
        self.features = nn.Sequential(conv_bn_act(3, 32, stride=2), DepthwiseSeparableConv(32, 64), nn.MaxPool2d(kernel_size=2, stride=2), DepthwiseSeparableConv(64, 128), DepthwiseSeparableConv(128, 128), nn.MaxPool2d(kernel_size=2, stride=2), DepthwiseSeparableConv(128, 256), DepthwiseSeparableConv(256, 256), nn.MaxPool2d(kernel_size=2, stride=2), DepthwiseSeparableConv(256, 512), DepthwiseSeparableConv(512, 512), DepthwiseSeparableConv(512, 512), DepthwiseSeparableConv(512, 512), DepthwiseSeparableConv(512, 512), DepthwiseSeparableConv(512, 512), nn.MaxPool2d(kernel_size=2, stride=2), DepthwiseSeparableConv(512, 1024), DepthwiseSeparableConv(1024, 1024), nn.AdaptiveAvgPool2d((1, 1))).to(device)
        self.backbone_a = TorchVision('mobilenet_v3_small', in_channels=3).to(device)
        self.backbone_b = TorchVision('squeezenet1_1', in_channels=3).to(device)
        self.infer_dimensions_dynamically(in_shape, out_shape[0])
        self._scaler = make_scaler(enabled=self.use_amp)

    def infer_dimensions_dynamically(self, in_shape, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            C = in_shape[1] if len(in_shape) == 4 else in_shape[0]
            dummy = torch.zeros(1, C, 224, 224).to(self.device)
            output_feat = self.forward(dummy, is_probing=True)
            dim_fused = output_feat.shape[1]
        self.classifier = nn.Linear(dim_fused, num_classes)
        self.train()

    def forward(self, x: torch.Tensor, is_probing: bool=False) -> torch.Tensor:
        x = x.to(self.device)
        x_f = adaptive_pool_flatten(self.features(x))
        x_a = adaptive_pool_flatten(self.backbone_a(x))
        x_b = adaptive_pool_flatten(self.backbone_b(x))
        fused = torch.cat([x_f, x_a, x_b], dim=1)
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