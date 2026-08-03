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

def depthwise_separable_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
    return nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, groups=in_channels, bias=bias), nn.LazyConv2d(out_channels, kernel_size=1, stride=1, bias=bias), nn.LazyBatchNorm2d(), nn.ReLU(inplace=True), nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity())

def inverted_residual_block(in_channels, out_channels, stride=1, expand_ratio=6, dropout_prob=0.0):
    hidden_dim = round(in_channels * expand_ratio)
    layers = [nn.LazyConv2d(hidden_dim, kernel_size=1, bias=False), nn.LazyBatchNorm2d(), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False), nn.LazyBatchNorm2d(), nn.ReLU6(inplace=True), nn.LazyConv2d(out_channels, kernel_size=1, bias=False), nn.LazyBatchNorm2d(), nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()]
    if stride != 1 or in_channels != out_channels:
        layers.insert(0, nn.LazyConv2d(out_channels, kernel_size=1, stride=stride, bias=False))
        layers.insert(1, nn.LazyBatchNorm2d())
    return nn.Sequential(*layers)

def squeeze_excitation_block(in_channels, reduction_ratio=16):
    reduced_dim = in_channels // reduction_ratio
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.LazyConv2d(reduced_dim, kernel_size=1), nn.ReLU(inplace=True), nn.LazyConv2d(in_channels, kernel_size=1), nn.Sigmoid())

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.use_amp = prm.get('use_amp', False)
        dropout_prob = float(prm['dropout'])
        self.backbone_a = TorchVision('mobilenet_v3_small', in_channels=3).to(device)
        self.backbone_b = TorchVision('shufflenet_v2_x1_0', in_channels=3).to(device)
        self.features = nn.Sequential(inverted_residual_block(in_channels=3, out_channels=24, stride=1, expand_ratio=6, dropout_prob=dropout_prob), squeeze_excitation_block(in_channels=24, reduction_ratio=16), depthwise_separable_block(in_channels=24, out_channels=48, stride=2, padding=1, bias=False, dropout_prob=dropout_prob), squeeze_excitation_block(in_channels=48, reduction_ratio=16), inverted_residual_block(in_channels=48, out_channels=96, stride=2, expand_ratio=6, dropout_prob=dropout_prob), squeeze_excitation_block(in_channels=96, reduction_ratio=16))
        self.infer_dimensions_dynamically(in_shape, out_shape[0])
        self.classifier = nn.LazyLinear(out_shape[0])
        self.to(self.device)
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