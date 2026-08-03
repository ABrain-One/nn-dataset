import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, groups=in_channels, bias=False)
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
        super(InvertedResidualBlock, self).__init__()
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
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, reduced_dim, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True), nn.Conv2d(reduced_dim, in_channels, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, x):
        se_weights = self.se(x)
        return x * se_weights

class CustomMobileNetV3Small(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(CustomMobileNetV3Small, self).__init__()
        self.device = device
        self.to(self.device)
        dummy = torch.zeros(1, in_shape[1], 256, 256).to(self.device)
        base_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        base_model.classifier = nn.Identity()
        self.base = base_model
        self.fc = nn.Linear(576, out_shape[0])

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def infer_dimensions_dynamically(self):
        dummy = torch.zeros(1, self.in_shape[1], 256, 256).to(self.device)
        with torch.no_grad():
            output = self.forward(dummy)
        return output.shape[1]

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

class Net(CustomMobileNetV3Small):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__(in_shape, out_shape, prm, device)