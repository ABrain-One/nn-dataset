import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision
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

def ds_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
    return nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, groups=in_channels, bias=bias), nn.BatchNorm2d(in_channels), nn.ReLU6(inplace=True), nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias), nn.Dropout2d(p=dropout_prob))

class InvertedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expand_ratio, dropout_prob=0.0):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        hidden_dim = round(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        layers = []
        if expand_ratio != 1:
            layers.append(ds_conv3x3_block(in_channels, hidden_dim, bias=False, dropout_prob=dropout_prob))
        layers.append(ds_conv3x3_block(hidden_dim, out_channels, stride=stride, bias=False, dropout_prob=dropout_prob))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class SqueezeExcitation(nn.Module):

    def __init__(self, channels, reduction_ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        se_weights = self.se(x)
        return x * se_weights

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.use_amp = prm.get('use_amp', False)
        dropout_prob = float(prm['dropout'])
        self.backbone_a = TorchVision('mobilenet_v3_small', in_channels=3).to(device)
        self.features = nn.Sequential(ds_conv3x3_block(3, 16, stride=1, padding=1, bias=False, dropout_prob=dropout_prob), InvertedResidualBlock(16, 24, stride=1, expand_ratio=2, dropout_prob=dropout_prob), InvertedResidualBlock(24, 24, stride=2, expand_ratio=1, dropout_prob=dropout_prob), InvertedResidualBlock(24, 40, stride=1, expand_ratio=2, dropout_prob=dropout_prob), InvertedResidualBlock(40, 40, stride=2, expand_ratio=1, dropout_prob=dropout_prob), SqueezeExcitation(40, 10), nn.MaxPool2d(kernel_size=2, stride=2))
        self.infer_dimensions_dynamically(in_shape, out_shape[0])
        self.classifier = nn.Linear(self.fused_dim, out_shape[0]).to(device)
        self._scaler = GradScaler('cuda', enabled=self.use_amp)
        self.to(self.device)

    def infer_dimensions_dynamically(self, in_shape, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            C = in_shape[1] if len(in_shape) == 4 else in_shape[0]
            dummy = torch.zeros(1, C, 32, 32).to(self.device)
            output_feat = self.forward(dummy, is_probing=True)
            self.fused_dim = output_feat.shape[1]
        self.train()

    def forward(self, x: torch.Tensor, is_probing: bool=False) -> torch.Tensor:
        x = x.to(self.device)
        x_f = adaptive_pool_flatten(self.features(x))
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