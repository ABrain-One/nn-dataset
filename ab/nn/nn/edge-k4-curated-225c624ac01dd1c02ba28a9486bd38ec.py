import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision
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

def depthwise_separable_block(in_channels, out_channels, stride=1, activation=nn.ReLU(inplace=True)):
    return nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False), nn.BatchNorm2d(in_channels), activation, nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out_channels), activation)

class SEBlock(nn.Module):

    def __init__(self, in_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels // reduction_ratio, bias=False), nn.ReLU(inplace=True), nn.Linear(in_channels // reduction_ratio, in_channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InvertedResidualSEBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, expansion_factor=4):
        super(InvertedResidualSEBlock, self).__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expansion_factor)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        layers = []
        if in_channels != hidden_dim:
            layers.append(depthwise_separable_block(in_channels, hidden_dim, stride=1))
        layers.extend([nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=self.stride, padding=1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), SEBlock(hidden_dim), nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out_channels)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.pattern = 'Parallel_Dual_SE'
        self.use_amp = prm.get('use_amp', False)
        dropout_prob = float(prm.get('dropout', 0.2))
        self.backbone_a = TorchVision('mobilenet_v3_small', in_channels=3).to(device)
        self.features = nn.Sequential()
        curr_ch = 3
        channels = [16, 24, 32, 64, 96, 160]
        for i, out_ch in enumerate(channels):
            self.features.add_module(f'unit{i + 1}', InvertedResidualSEBlock(curr_ch, out_ch, stride=1 if i == 0 else 2))
            curr_ch = out_ch
        self.infer_dimensions_dynamically(in_shape, out_shape[0])
        self.dropout = nn.Dropout(dropout_prob)
        self._scaler = GradScaler('cuda', enabled=self.use_amp)

    def infer_dimensions_dynamically(self, in_shape, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            C = in_shape[1] if len(in_shape) == 4 else in_shape[0]
            dummy = torch.zeros(1, C, 32, 32).to(self.device)
            output_feat = self.forward(dummy, is_probing=True)
            dim_fused = output_feat.shape[1]
        self.classifier = nn.Linear(dim_fused, num_classes)
        self.train()

    def _init_params(self):
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _norm4d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        raise ValueError(f'Expected 4D/5D input, got {tuple(x.shape)}')

    def forward(self, x: torch.Tensor, is_probing: bool=False) -> torch.Tensor:
        x = self._norm4d(x).to(self.device)
        x_f = adaptive_pool_flatten(self.features(x))
        x_a = adaptive_pool_flatten(self.backbone_a(x))
        fused = torch.cat([x_f, x_a], dim=1)
        if is_probing:
            return fused
        fused = self.dropout(fused)
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