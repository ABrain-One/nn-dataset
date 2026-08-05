import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import mobilenet_v3_small

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InvertedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = round(in_channels * expand_ratio)
        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(SqueezeExcitation(hidden_dim, hidden_dim // 4))
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.body(x)
        else:
            return self.body(x)

class SqueezeExcitation(nn.Module):

    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, reduced_dim, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True), nn.Conv2d(reduced_dim, in_channels, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.to(self.device)
        self.infer_dimensions_dynamically(in_shape, out_shape, prm)

    def infer_dimensions_dynamically(self, in_shape, out_shape, prm):
        dummy = torch.zeros(1, in_shape[1], 256, 256).to(self.device)
        base_model = mobilenet_v3_small(weights='DEFAULT')
        base_model.classifier = nn.Identity()
        self.backbone = base_model.features.to(self.device)
        dummy = self.backbone(dummy)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        dummy = self.global_avg_pool(dummy)
        flattened_size = dummy.view(1, -1).size(1)
        self.classifier = nn.Linear(flattened_size, out_shape[0]).to(self.device)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self._batch_size = int(prm['batch'])
        self._epochs = int(prm['epoch'])
        self._dropout = float(prm['dropout'])
        self._transform = prm['transform']
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()