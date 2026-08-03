import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision
from torch.nn import MaxPool2d
from torch.amp import autocast, GradScaler

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, expansion_factor=6, stride=1):
        super().__init__()
        self.expansion = expansion_factor
        expanded_dim = in_channels * self.expansion
        self.residual_connection = stride == 1 and in_channels == out_channels
        self.expand = nn.Conv2d(in_channels, expanded_dim, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(expanded_dim, expanded_dim, kernel_size=3, stride=stride, padding=1, groups=expanded_dim, bias=False)
        self.bn = nn.BatchNorm2d(expanded_dim)
        self.squeeze_excite = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(expanded_dim, expanded_dim // 16, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(expanded_dim // 16, expanded_dim, kernel_size=1), nn.Sigmoid())
        self.project = nn.Conv2d(expanded_dim, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x_se = self.squeeze_excite(x)
        x = x * x_se
        x = self.project(x)
        if self.residual_connection:
            x += identity
        return x

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.to(self.device)
        self.channels = in_shape[1]
        self.classes = out_shape[0]
        self.dropout_rate = prm['dropout']
        self.base_channels = 32
        self.first_conv = nn.Sequential(nn.Conv2d(self.channels, self.base_channels, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(self.base_channels), nn.ReLU(inplace=True))
        self.bottlenecks = nn.Sequential(InvertedResidual(self.base_channels, 16, stride=1), InvertedResidual(16, 24, stride=2), InvertedResidual(24, 24, stride=1), InvertedResidual(24, 32, stride=2), InvertedResidual(32, 32, stride=1), InvertedResidual(32, 32, stride=1), InvertedResidual(32, 64, stride=2), InvertedResidual(64, 64, stride=1), InvertedResidual(64, 64, stride=1), InvertedResidual(64, 64, stride=1), InvertedResidual(64, 96, stride=2), InvertedResidual(96, 96, stride=1), InvertedResidual(96, 96, stride=1), InvertedResidual(96, 160, stride=2), InvertedResidual(160, 160, stride=1), InvertedResidual(160, 160, stride=1), InvertedResidual(160, 320, stride=1), InvertedResidual(320, 320, stride=1))
        self.final_conv = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, self.classes)
        self.drop_out = nn.Dropout(self.dropout_rate)
        self.infer_dimensions_dynamically(in_shape, self.classes)

    def infer_dimensions_dynamically(self, in_shape, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            C = in_shape[1] if len(in_shape) == 4 else in_shape[0]
            dummy = torch.zeros(1, C, 32, 32).to(self.device)
            output = self.forward(dummy, is_probing=True)
            dim_fused = output.shape[1]
        self.classifier = nn.Linear(dim_fused, num_classes)
        self.train()

    def forward(self, x, is_probing=False):
        x = self.first_conv(x)
        x = self.bottlenecks(x)
        x = self.final_conv(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        if is_probing:
            return x
        x = self.drop_out(x)
        x = self.classifier(x)
        return x

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
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()