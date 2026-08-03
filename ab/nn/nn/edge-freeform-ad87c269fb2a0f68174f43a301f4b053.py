import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision
from torch.nn import MaxPool2d
import torch.nn.functional as F

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

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.relu = nn.ReLU6(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class InvertedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, expansion_factor, stride, dropout_prob=0.0):
        super().__init__()
        expanded_channels = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels
        self.conv1 = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)
        self.dwconv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(expanded_channels)
        self.conv2 = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dwconv(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.dropout(x)
        if self.use_residual:
            x += identity
        return x

class SqueezeExcitation(nn.Module):

    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels, reduced_dim, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(reduced_dim, in_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        scale = self.se(x)
        return x * scale

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.use_amp = False
        dropout_prob = float(prm['dropout'])
        self.backbone_a = TorchVision('mobilenet_v3_small', in_channels=3).to(device)
        self.features = nn.Sequential()
        curr_ch = 1024
        channels = [128, 256, 512]
        for i, out_ch in enumerate(channels):
            if i == 0:
                self.features.add_module(f'unit{i + 1}_ds', DepthwiseSeparableConv(curr_ch, out_ch, stride=2, dropout_prob=dropout_prob))
            else:
                self.features.add_module(f'unit{i + 1}_irb1', InvertedResidualBlock(curr_ch, out_ch, expansion_factor=6, stride=2, dropout_prob=dropout_prob))
                self.features.add_module(f'unit{i + 1}_se', SqueezeExcitation(out_ch, out_ch // 4))
                self.features.add_module(f'unit{i + 1}_irb2', InvertedResidualBlock(out_ch, out_ch, expansion_factor=6, stride=1, dropout_prob=dropout_prob))
                self.features.add_module(f'unit{i + 1}_se2', SqueezeExcitation(out_ch, out_ch // 4))
            curr_ch = out_ch
        self.backbone_b = TorchVision('shufflenet_v2_x1_0', in_channels=3).to(device)
        self.infer_dimensions_dynamically(in_shape, out_shape[0])

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
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
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
        x_bf = self.backbone_b(x)
        if x_bf.dim() == 2:
            x_bf = x_bf.unsqueeze(-1).unsqueeze(-1)
        if x_bf.shape[-1] < 14:
            x_bf = torch.nn.functional.interpolate(x_bf, size=(14, 14), mode='nearest')
        f_bf = adaptive_pool_flatten(self.features(x_bf))
        fused = torch.cat([f_a, f_bf], dim=1)
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

    def learn(self, train_data):
        self.train()
        train_iter = iter(train_data)
        try:
            for batch_idx, (inputs, labels) in enumerate(train_iter):
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                self.optimizer.step()
        finally:
            if hasattr(train_iter, 'shutdown'):
                train_iter.shutdown()
            del train_iter
            gc.collect()