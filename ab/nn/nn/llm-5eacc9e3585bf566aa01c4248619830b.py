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
        kwargs = {'aux_logits': False} if 'inception' in model.lower() else {}
        try:
            if hasattr(torchvision.models, 'get_model'):
                self.m = torchvision.models.get_model(model, weights=weights, **kwargs)
            else:
                self.m = torchvision.models.__dict__[model](pretrained=bool(weights), **kwargs)
        except:
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
    return {'lr', 'dropout', 'momentum'}

class EfficientConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        self.skip_bn = nn.BatchNorm2d(out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        identity = self.skip_bn(identity)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + identity
        out = self.act2(out)
        return out

class WideBackbone(nn.Module):

    def __init__(self, in_channels, dropout_prob=0.0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.stage1 = nn.Sequential(EfficientConvBlock(64, 128, dropout_prob), nn.MaxPool2d(kernel_size=2, stride=2), EfficientConvBlock(128, 128, dropout_prob))
        self.stage2 = nn.Sequential(EfficientConvBlock(128, 256, dropout_prob), nn.MaxPool2d(kernel_size=2, stride=2), EfficientConvBlock(256, 256, dropout_prob))
        self.stage3 = nn.Sequential(EfficientConvBlock(256, 512, dropout_prob), nn.MaxPool2d(kernel_size=2, stride=2))
        self.stage4 = nn.Sequential(EfficientConvBlock(512, 1024, dropout_prob))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.use_amp = prm.get('use_amp', False)
        c_in = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.backbone = TorchVision(model='resnet18', in_channels=c_in, truncate=1)
        self.custom_head = WideBackbone(in_channels=c_in, dropout_prob=prm.get('dropout', 0.0))
        self._input_spec = (c_in, in_shape[-2], in_shape[-1])
        self.infer_dimensions_dynamically(out_shape[0])
        self._scaler = make_scaler(enabled=self.use_amp)

    def infer_dimensions_dynamically(self, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            c, h, w = self._input_spec
            dummy = torch.zeros(1, c, h, w).to(self.device)
            feat_backbone = adaptive_pool_flatten(self.backbone(dummy))
            feat_custom = adaptive_pool_flatten(self.custom_head(dummy))
            fused = torch.cat([feat_backbone, feat_custom], dim=1)
            dim_fused = fused.shape[1]
        self.classifier = nn.Sequential(nn.Linear(dim_fused, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, num_classes))
        self.train()

    @staticmethod
    def _norm4d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        raise ValueError(f'Expected 4D/5D input, got {tuple(x.shape)}')

    def forward(self, x: torch.Tensor, is_probing: bool=False) -> torch.Tensor:
        x_backbone = adaptive_pool_flatten(self.backbone(x))
        x_custom = adaptive_pool_flatten(self.custom_head(x))
        fused = torch.cat([x_backbone, x_custom], dim=1)
        if is_probing:
            return fused
        return self.classifier(fused)

    def train_setup(self, prm):
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