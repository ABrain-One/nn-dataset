import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=in_shape[1], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

# Because of the edge-friendly latency constraints, we'll use a small model with depthwise separable convolutions
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, stride=stride)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.batch_norm(x)
        return self.relu(x)

    def fuse_conv_bn(self):
        fused = nn.Conv2d(self.depthwise.in_channels, self.pointwise.out_channels, kernel_size=self.depthwise.kernel_size,
                          stride=self.depthwise.stride, padding=self.depthwise.padding, groups=self.depthwise.groups)
        fused.weight.data = self.depthwise.weight.data * self.pointwise.weight.data
        if self.pointwise.bias is not None:
            fused.bias.data = self.pointwise.bias.data
        fused.requires_grad = False
        return fused

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=in_shape[1], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.separable_conv1 = SeparableConvBlock(32, 64, stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.separable_conv2 = SeparableConvBlock(64, 128, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        
        x = self.separable_conv1(x)
        x = self.max_pool1(x)
        
        x = self.separable_conv2(x)
        x = self.max_pool2(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    # Ensure SeparableConvBlock is defined above

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)