import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v3_small, shufflenet_v2_x1_0
from torchvision.models._utils import _make_divisible
from torchvision.ops.misc import Conv2dNormActivation
import torchvision.transforms as transforms

def supported_hyperparameters():
    return {'dropout', 'lr', 'momentum'}

class InvertedResidualSE(nn.Module):

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int, se_ratio: float=0.25):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(Conv2dNormActivation(inp, hidden_dim, kernel_size=1, activation_layer=nn.ReLU6))
        layers.extend([Conv2dNormActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, activation_layer=nn.ReLU6), SqueezeExcitation(hidden_dim, hidden_dim, se_ratio), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: int, expanded_channels: int, se_ratio: float):
        super().__init__()
        reduced_dim = _make_divisible(input_channels * se_ratio, 8)
        self.fc1 = nn.Conv2d(expanded_channels, reduced_dim, 1)
        self.fc2 = nn.Conv2d(reduced_dim, expanded_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean([2, 3], keepdim=True)
        scale = self.fc1(scale)
        scale = F.relu(scale)
        scale = self.fc2(scale)
        scale = torch.sigmoid(scale)
        return scale * x

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.to(self.device)
        self.infer_dimensions_dynamically(in_shape, out_shape, prm)

    def infer_dimensions_dynamically(self, in_shape: tuple, out_shape: tuple, prm: dict):
        dummy = torch.zeros(1, in_shape[1], 256, 256).to(self.device)
        backbone = mobilenet_v3_small(pretrained=False, weights=None)
        backbone.features[0][0] = nn.Conv2d(in_shape[1], 16, kernel_size=3, stride=2, padding=1, bias=False)
        backbone.classifier[-1] = nn.Linear(backbone.classifier[-1].in_features, out_shape[0])
        self.features = backbone.features
        self.classifier = backbone.classifier
        self.dropout = nn.Dropout(prm['dropout'])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = out_shape[0]
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

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