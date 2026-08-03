import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision
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

class DepthwiseSeparableBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, expansion_factor=6, dropout_prob=0.0):
        super().__init__()
        self.expand = nn.LazyConv2d(in_channels * expansion_factor, kernel_size=1, bias=False)
        self.bn_expand = nn.LazyBatchNorm2d()
        self.depthwise = nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor, kernel_size=3, stride=stride, padding=1, groups=in_channels * expansion_factor, bias=False)
        self.bn_depthwise = nn.LazyBatchNorm2d()
        self.project = nn.LazyConv2d(out_channels, kernel_size=1, bias=False)
        self.bn_project = nn.LazyBatchNorm2d()
        self.relu = nn.ReLU6(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.LazyConv2d(out_channels, kernel_size=1, stride=stride, bias=False), nn.LazyBatchNorm2d())
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.bn_expand(x)
        x = self.relu(x)
        x = self.depthwise(x)
        x = self.bn_depthwise(x)
        x = self.relu(x)
        x = self.project(x)
        x = self.bn_project(x)
        x = self.dropout(x)
        x += self.shortcut(identity)
        return self.relu(x)

class InvertedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion_factor, dropout_prob=0.0):
        super().__init__()
        self.stride = stride
        hidden_dim = round(in_channels * expansion_factor)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        layers = [nn.LazyConv2d(hidden_dim, kernel_size=1, bias=False), nn.LazyBatchNorm2d(), nn.ReLU6(inplace=True)]
        layers.extend([nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False), nn.LazyBatchNorm2d(), nn.ReLU6(inplace=True)])
        layers.extend([nn.LazyConv2d(out_channels, kernel_size=1, bias=False), nn.LazyBatchNorm2d()])
        self.conv = nn.Sequential(*layers)
        self.se = SqueezeExcitationBlock(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = self.se(x)
        x = self.dropout(x)
        return x + identity if self.use_res_connect else x

class SqueezeExcitationBlock(nn.Module):

    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.LazyConv2d(channels // reduction_ratio, kernel_size=1, bias=True), nn.ReLU(inplace=True), nn.LazyConv2d(channels, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, x):
        scale = self.fc(x)
        return x * scale

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.use_amp = prm.get('use_amp', False)
        dropout_prob = float(prm['dropout'])
        self.backbone_a = TorchVision('shufflenet_v2_x1_0', in_channels=in_shape[1]).to(device)
        self.backbone_b = TorchVision('mobilenet_v3_small', in_channels=in_shape[1]).to(device)
        self.features = nn.Sequential()
        curr_ch = 576
        channels = [256, 128]
        strides = [2, 2]
        for i, (out_ch, stride) in enumerate(zip(channels, strides)):
            self.features.add_module(f'ds_unit{i + 1}', DepthwiseSeparableBlock(curr_ch, out_ch, stride=stride, expansion_factor=6, dropout_prob=dropout_prob))
            self.features.add_module(f'inverted_residual_unit{i + 1}', InvertedResidualBlock(out_ch, expansion_factor=4, out_channels=out_ch, stride=1, dropout_prob=dropout_prob))
            self.features.add_module(f'se_unit{i + 1}', SqueezeExcitationBlock(out_ch))
            curr_ch = out_ch
        self.infer_dimensions_dynamically(in_shape, out_shape[0])
        self.classifier = nn.LazyLinear(out_shape[0])
        self.to(self.device)
        self._scaler = GradScaler('cuda', enabled=self.use_amp)

    def infer_dimensions_dynamically(self, in_shape, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            C = in_shape[1] if len(in_shape) == 4 else in_shape[0]
            dummy = torch.zeros(1, C, 32, 32).to(self.device)
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
        mid_img = torch.nn.functional.interpolate(mid_4d, size=(16, 16), mode='nearest')
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