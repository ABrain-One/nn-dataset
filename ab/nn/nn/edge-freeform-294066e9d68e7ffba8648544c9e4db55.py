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
    return nn.Sequential(nn.LazyConv2d(out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias), nn.LazyBatchNorm2d(), nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity(), nn.SiLU(inplace=True))

class InvertedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, expansion_factor=4, stride=1, dropout_prob=0.0):
        super().__init__()
        self.stride = stride
        expanded_channels = in_channels * expansion_factor
        self.expand = drop_conv3x3_block(in_channels, expanded_channels, stride=1)
        self.depthwise = nn.Sequential(nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False), nn.BatchNorm2d(expanded_channels), nn.SiLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_channels))
        self.residual = nn.Identity() if stride == 1 and in_channels == out_channels else None
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        if self.residual:
            x += self.residual(identity)
        x = self.dropout(x)
        return x

class SEBlock(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        reduced_channels = max(in_channels // reduction_ratio, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_channels, reduced_channels), nn.ReLU(inplace=True), nn.Linear(reduced_channels, in_channels), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        se_weights = self.avg_pool(x).view(b, c)
        se_weights = self.fc(se_weights).view(b, c, 1, 1)
        return x * se_weights.expand_as(x)

class EfficientEdgeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, expansion_factor=4, stride=1, dropout_prob=0.0, se_ratio=4):
        super().__init__()
        self.irb = InvertedResidualBlock(in_channels, out_channels, expansion_factor, stride, dropout_prob)
        self.se = SEBlock(out_channels, se_ratio)

    def forward(self, x):
        x = self.irb(x)
        x = self.se(x)
        return x

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.pattern = 'Efficient_Edge_Network'
        self.use_amp = prm.get('use_amp', False)
        dropout_prob = float(prm['dropout'])
        self.backbone_a = TorchVision('shufflenet_v2_x0_5', in_channels=3).to(device)
        self.backbone_b = TorchVision('mobilenet_v3_small', in_channels=3).to(device)
        self.features = nn.Sequential()
        curr_ch = 576 + 2304
        channels = [64 * 2 ** i for i in range(1)]
        for i, out_ch in enumerate(channels):
            self.features.add_module(f'unit{i + 1}', EfficientEdgeBlock(curr_ch, out_ch, dropout_prob=dropout_prob))
            curr_ch = out_ch
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
        mid = torch.cat([f_a, f_b], dim=1)
        mid_4d = mid.unsqueeze(-1).unsqueeze(-1)
        mid_img = torch.nn.functional.interpolate(mid_4d, size=(14, 14), mode='nearest')
        fused = adaptive_pool_flatten(self.features(mid_img))
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