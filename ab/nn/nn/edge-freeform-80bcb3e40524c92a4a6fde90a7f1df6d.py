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

def depthwise_separable_conv(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU6(inplace=True), nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))

class InvertedResidual(nn.Module):

    def __init__(self, in_channels, expansion_factor, out_channels, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expanded_dim = expansion_factor * in_channels
        self.conv = nn.Sequential(depthwise_separable_conv(in_channels, self.expanded_dim, 1), depthwise_separable_conv(self.expanded_dim, out_channels, stride))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

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
        self.to(self.device)
        dropout_prob = float(prm['dropout'])
        self.backbone_a = TorchVision('shufflenet_v2_x1_0', in_channels=in_shape[1]).to(device)
        self.features = nn.Sequential(InvertedResidual(in_shape[1], 2, 32, 1), SqueezeExcitation(32), InvertedResidual(32, 4, 64, 2), SqueezeExcitation(64), InvertedResidual(64, 4, 128, 2), SqueezeExcitation(128), nn.Dropout2d(dropout_prob), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.infer_dimensions_dynamically(in_shape, out_shape[0])
        self._scaler = GradScaler('cuda', enabled=False)

    def infer_dimensions_dynamically(self, in_shape, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            C = in_shape[1] if len(in_shape) == 4 else in_shape[0]
            dummy = torch.zeros(1, C, 32, 32).to(self.device)
            x_f = adaptive_pool_flatten(self.features(dummy))
            x_a = adaptive_pool_flatten(self.backbone_a(dummy))
            dim_fused = x_f.size(1) + x_a.size(1)
        self.classifier = nn.Linear(dim_fused, num_classes)
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

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()