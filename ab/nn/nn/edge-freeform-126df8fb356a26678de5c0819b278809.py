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

class SqueezeExcitation(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class InvertedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, expansion_factor=4, stride=1):
        super().__init__()
        expanded_channels = in_channels * expansion_factor
        self.conv1 = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparableConv(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1)
        self.se = SqueezeExcitation(expanded_channels)
        self.conv3 = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.residual_connection = stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.residual_connection:
            x += identity
        x = self.relu1(x)
        return x

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.to(self.device)
        self.infer_dimensions_dynamically(in_shape, out_shape, prm)

    def infer_dimensions_dynamically(self, in_shape, out_shape, prm):
        dummy = torch.zeros(1, in_shape[1], in_shape[2], in_shape[3]).to(self.device)
        self.backbone = models.mobilenet_v3_small(pretrained=False, weights='DEFAULT').features.to(self.device)
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