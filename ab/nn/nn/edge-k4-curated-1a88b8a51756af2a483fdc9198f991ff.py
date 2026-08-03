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

def ds_conv_block(in_channels, out_channels, stride=1, expansion_factor=1):
    hidden_dim = round(expansion_factor * in_channels)
    return nn.Sequential(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True), nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_channels))

class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(expansion_factor * in_channels)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        layers = []
        if expansion_factor != 1:
            layers.append(ds_conv_block(in_channels, hidden_dim, 1, expansion_factor))
        layers.extend([ds_conv_block(hidden_dim, hidden_dim, stride, 1), nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_channels)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.use_amp = prm.get('use_amp', False)
        dropout_prob = float(prm.get('dropout', 0.1))
        self.backbone = TorchVision('efficientnet_b0', in_channels=in_shape[1]).to(device)
        self.features = nn.Sequential(InvertedResidual(1280, 256, 1, 2), InvertedResidual(256, 256, 1, 1), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(dropout_prob), nn.Linear(256, 128), nn.SiLU(inplace=True))
        self.classifier = nn.Linear(128, out_shape[0])
        self.infer_dimensions_dynamically(in_shape, out_shape[0])
        self._scaler = GradScaler('cuda', enabled=self.use_amp)

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
        feat = self.backbone(x)
        feat = self.features(feat)
        if is_probing:
            return feat
        return self.classifier(feat)

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