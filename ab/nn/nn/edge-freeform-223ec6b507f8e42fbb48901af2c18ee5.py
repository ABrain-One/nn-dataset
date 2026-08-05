import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import math

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

class SEBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InvertedResidualSE(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualSE, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), SEBlock(hidden_dim), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), SEBlock(hidden_dim), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.to(self.device)
        self.backbone = models.mobilenet_v3_small(pretrained=False, weights=None)
        self.backbone.classifier = nn.Sequential()
        self.feature_extractor = nn.Sequential(nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2, padding=1), InvertedResidualSE(32, 16, 1, 1), InvertedResidualSE(16, 24, 2, 6), InvertedResidualSE(24, 24, 1, 6), InvertedResidualSE(24, 40, 2, 6), InvertedResidualSE(40, 40, 1, 6), InvertedResidualSE(40, 40, 1, 6), InvertedResidualSE(40, 80, 2, 6), InvertedResidualSE(80, 80, 1, 6), InvertedResidualSE(80, 80, 1, 6), InvertedResidualSE(80, 80, 1, 6), InvertedResidualSE(80, 112, 1, 6), InvertedResidualSE(112, 112, 1, 6), InvertedResidualSE(112, 160, 2, 6), InvertedResidualSE(160, 160, 1, 6), InvertedResidualSE(160, 160, 1, 6), InvertedResidualSE(160, 96, 1, 6), nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(start_dim=1), nn.Dropout(p=prm['dropout']), nn.Linear(96, out_shape[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return x

    def train_setup(self, prm):
        self._batch_size = int(prm['batch'])
        self._epochs = int(prm['epoch'])
        self._transform = prm['transform']
        self.to(self.device)
        learning_rate = float(prm['lr'])
        momentum = float(prm['momentum'])
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(next(self.parameters()).device), labels.to(next(self.parameters()).device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()