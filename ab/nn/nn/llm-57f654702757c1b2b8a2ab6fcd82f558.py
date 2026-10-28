import torch
import torch.nn.functional as F
from typing import Any
from types import FunctionType

def _log_api_usage_once(obj: Any) -> None:
    module = obj.__module__
    if not module.startswith('torchvision'):
        module = f'torchvision.internal.{module}'
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f'{module}.{name}')

class ChannelOptimized(torch.nn.Module):

    def __init__(self, threshold, p):
        super().__init__()
        _log_api_usage_once(self)
        self.threshold = threshold
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            return torch.where(img >= self.threshold, 1.0 - img, img)
        return img

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(threshold={self.threshold},p={self.p})'

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(torch.nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.classifier = torch.nn.Linear(1024, self.num_classes)

    def build_features(self):
        layers = []
        layers += [torch.nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(inplace=True), torch.nn.MaxPool2d(kernel_size=2, stride=2), torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(inplace=True), torch.nn.MaxPool2d(kernel_size=2, stride=2), torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True), torch.nn.MaxPool2d(kernel_size=2, stride=2), torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True), torch.nn.MaxPool2d(kernel_size=2, stride=2), torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), torch.nn.BatchNorm2d(1024), torch.nn.ReLU(inplace=True)]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = torch.nn.CrossEntropyLoss()

    def learn(self, data_roll):
        for data, target in data_roll:
            data, target = (data.to(self.device), target.to(self.device))
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()