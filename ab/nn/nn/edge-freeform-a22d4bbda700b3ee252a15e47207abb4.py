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

def inverted_residual_block(in_channels, out_channels, stride=1, expansion_factor=6, dropout_prob=0.0):
    hidden_dim = int(round(in_channels * expansion_factor))
    layers = [nn.LazyConv2d(hidden_dim, kernel_size=1, stride=1, padding=0, bias=False), nn.LazyBatchNorm2d(), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False), nn.LazyBatchNorm2d(), nn.ReLU6(inplace=True), nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity(), nn.LazyConv2d(out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.LazyBatchNorm2d()]
    if stride != 1 or in_channels != out_channels:
        layers.insert(0, nn.LazyConv2d(out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
        layers.insert(1, nn.LazyBatchNorm2d())
    return nn.Sequential(*layers)

class SEBlock(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.LazyLinear(in_channels // reduction_ratio, bias=False), nn.ReLU(inplace=True), nn.LazyLinear(in_channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.pointwise(self.depthwise(x)))))

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.use_amp = prm.get('use_amp', False)
        dropout_prob = float(prm.get('dropout', 0.1))
        self.backbone_a = TorchVision('mobilenet_v3_small', in_channels=3).to(device)
        self.backbone_b = TorchVision('squeezenet1_1', in_channels=3).to(device)
        self.features = nn.Sequential(inverted_residual_block(3, 16, stride=1, expansion_factor=6, dropout_prob=dropout_prob), SEBlock(16), DepthwiseSeparableBlock(16, 24, stride=2, dropout_prob=dropout_prob), inverted_residual_block(24, 40, stride=2, expansion_factor=6, dropout_prob=dropout_prob), SEBlock(40), DepthwiseSeparableBlock(40, 80, stride=2, dropout_prob=dropout_prob), inverted_residual_block(80, 96, stride=1, expansion_factor=6, dropout_prob=dropout_prob), SEBlock(96), DepthwiseSeparableBlock(96, 192, stride=2, dropout_prob=dropout_prob), inverted_residual_block(192, 320, stride=1, expansion_factor=6, dropout_prob=dropout_prob), SEBlock(320)).to(device)
        self.infer_dimensions_dynamically(in_shape, out_shape[0])
        self._scaler = GradScaler('cuda', enabled=self.use_amp)
        self.to(self.device)

    def infer_dimensions_dynamically(self, in_shape, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            C = in_shape[1] if len(in_shape) == 4 else in_shape[0]
            dummy = torch.zeros(1, C, 256, 256).to(self.device)
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
        f_a = adaptive_pool_flatten(self.backbone_a(x))
        f_b = adaptive_pool_flatten(self.backbone_b(x))
        f_f = adaptive_pool_flatten(self.features(x))
        fused = torch.cat([f_a, f_b, f_f], dim=1)
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