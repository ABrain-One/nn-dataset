import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision

def adaptive_pool_flatten(x):
    if x.ndim == 4:
        return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    if x.ndim == 3:
        return x.mean(dim=1)
    return x.flatten(1) if x.ndim > 2 else x

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class InvertedResidualBlock(nn.Module):

    def __init__(self, in_channels, expansion_factor, out_channels, stride):
        super(InvertedResidualBlock, self).__init__()
        expanded_channels = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False), nn.BatchNorm2d(expanded_channels), nn.ReLU6(inplace=True), nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False), nn.BatchNorm2d(expanded_channels), nn.ReLU6(inplace=True), nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        dropout_prob = float(prm['dropout'])
        self.backbone = torchvision.models.mobilenet_v3_small(weights='DEFAULT').features.to(device)
        self.features = nn.Sequential(InvertedResidualBlock(576, 6, 96, 1), nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity(), DepthwiseSeparableConv(96, 128, 1), InvertedResidualBlock(128, 6, 128, 2), DepthwiseSeparableConv(128, 128, 1), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity(), nn.Linear(128, out_shape[0]))
        self.infer_dimensions_dynamically(in_shape, out_shape[0])

    def infer_dimensions_dynamically(self, in_shape, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            C = in_shape[1]
            dummy = torch.zeros(1, C, 32, 32).to(self.device)
            output_feat = self.features(self.backbone(dummy))
            dim_fused = output_feat.shape[1]
        self.classifier = nn.Linear(dim_fused, num_classes)
        self.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.features(x)
        return self.classifier(x)

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