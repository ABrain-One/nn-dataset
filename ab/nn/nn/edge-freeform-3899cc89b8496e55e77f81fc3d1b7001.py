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
        self.adapter = nn.LazyConv2d(3, kernel_size=1) if in_channels != 3 else nn.Identity()
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

def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
    return nn.Sequential(nn.LazyConv2d(out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias), nn.LazyBatchNorm2d(), nn.ReLU(inplace=True), nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity(), nn.LazyConv2d(out_channels, kernel_size=3, stride=1, padding=1, bias=bias), nn.LazyBatchNorm2d(), nn.ReLU(inplace=True), nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity())

class InvertedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = in_channels * expansion_factor
        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers = [nn.LazyConv2d(hidden_dim, 1, 1, 0, bias=False), nn.LazyBatchNorm2d(), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.LazyBatchNorm2d(), nn.ReLU6(inplace=True), nn.LazyConv2d(out_channels, 1, 1, 0, bias=False), nn.LazyBatchNorm2d()]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class SqueezeExcitation(nn.Module):

    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.LazyLinear(reduced_dim, bias=False), nn.ReLU(inplace=True), nn.LazyLinear(in_channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EnhancedInvertedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion_factor=6, reduction_ratio=16):
        super(EnhancedInvertedResidualBlock, self).__init__()
        hidden_dim = in_channels * expansion_factor
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.conv = InvertedResidualBlock(in_channels, out_channels, stride, expansion_factor)
        self.se = SqueezeExcitation(out_channels, out_channels // reduction_ratio)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.se(self.conv(x))
        else:
            return self.se(self.conv(x))

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.pattern = 'Efficient_Mobile_EfficientSE'
        self.use_amp = prm.get('use_amp', False)
        dropout_prob = float(prm.get('dropout', 0.1))
        self.backbone_a = TorchVision('mobilenet_v3_large', in_channels=3, truncate=2).to(device)
        self.backbone_b = TorchVision('efficientnet_b0', in_channels=3, truncate=2).to(device)
        self.features = nn.Sequential(nn.LazyConv2d(256, kernel_size=1, stride=1, padding=0, bias=False), nn.LazyBatchNorm2d(), nn.ReLU6(inplace=True), EnhancedInvertedResidualBlock(256, 256, stride=1), EnhancedInvertedResidualBlock(256, 256, stride=1), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
        self.infer_dimensions_dynamically(in_shape, out_shape[0])
        self._scaler = GradScaler('cuda', enabled=self.use_amp)
        try:
            import torch as _edge_torch
            self.to(device)
            _edge_was_training = self.training
            self.eval()
            with _edge_torch.no_grad():
                self(_edge_torch.zeros((2, *tuple(in_shape)[1:]), device=device))
            if _edge_was_training:
                self.train()
        except Exception:
            pass
        _edge_lazy_materialized = True

    def infer_dimensions_dynamically(self, in_shape, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            C = in_shape[1] if len(in_shape) == 4 else in_shape[0]
            dummy = torch.zeros(1, C, 224, 224).to(self.device)
            output_feat = self.forward(dummy, is_probing=True)
            dim_fused = output_feat.shape[1]
        self.classifier = nn.LazyLinear(num_classes)
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
        fa = self.backbone_a(x)
        fb = self.backbone_b(x)
        fused = torch.cat([fa, fb], dim=1)
        features = self.features(fused)
        if is_probing:
            return features
        return self.classifier(features)

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