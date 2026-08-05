import torch
import torch.nn as nn
import torchvision.models as models

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
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

    def __init__(self, in_channels, out_channels, expansion_factor=4, stride=1):
        super().__init__()
        self.expanded_channels = in_channels * expansion_factor
        self.stride = stride
        self.expand = nn.Conv2d(in_channels, self.expanded_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.expand_bn = nn.BatchNorm2d(self.expanded_channels)
        self.depthwise = DepthwiseSeparableConv(self.expanded_channels, self.expanded_channels, kernel_size=3, stride=stride, padding=1)
        self.squeeze_excitation = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(self.expanded_channels, self.expanded_channels // 16, kernel_size=1, stride=1, padding=0, bias=False), nn.ReLU(inplace=True), nn.Conv2d(self.expanded_channels // 16, self.expanded_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sigmoid())
        self.project = nn.Conv2d(self.expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        self.residual_connection = self.stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.expand_bn(x)
        x = self.depthwise(x)
        x = self.squeeze_excitation(x) * x
        x = self.project(x)
        x = self.project_bn(x)
        if self.residual_connection:
            x += identity
        return x

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.to(self.device)
        self.infer_dimensions_dynamically(in_shape, out_shape, prm)

    def infer_dimensions_dynamically(self, in_shape, out_shape, prm):
        dummy = torch.zeros(1, in_shape[1], in_shape[2], in_shape[3]).to(self.device)
        self.features = nn.Sequential(nn.Conv2d(in_shape[1], 16, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True), InvertedResidualBlock(16, 24, stride=2), InvertedResidualBlock(24, 24), InvertedResidualBlock(24, 40, stride=2), InvertedResidualBlock(40, 40), InvertedResidualBlock(40, 40), InvertedResidualBlock(40, 80, stride=2), InvertedResidualBlock(80, 80), InvertedResidualBlock(80, 80), InvertedResidualBlock(80, 80), InvertedResidualBlock(80, 112, stride=1), InvertedResidualBlock(112, 112), InvertedResidualBlock(112, 160, stride=2), InvertedResidualBlock(160, 160), InvertedResidualBlock(160, 160)).to(self.device)
        dummy = self.features(dummy)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        dummy = self.global_avg_pool(dummy)
        flattened_size = dummy.view(1, -1).size(1)
        self.classifier = nn.Linear(flattened_size, out_shape[0]).to(self.device)

    def forward(self, x):
        x = self.features(x)
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