import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(
            Block(32, 32),
            Block(32, 32)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Block(64, 64),
            Block(64, 64)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer3 = nn.Sequential(
            Block(64, 128, kernel_size=3, stride=1, padding=1, groups=64),
            Block(128, 128, kernel_size=3, stride=1, padding=1, groups=128)
        )
        self.layer4 = nn.Sequential(
            Block(128, 256, kernel_size=3, stride=1, padding=1, groups=128),
            Block(256, 256, kernel_size=3, stride=1, padding=1, groups=256)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(self.layer1(x))
        x = self.layer2(x)
        x = self.maxpool(self.layer3(x))
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
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
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# Dummy input to determine input and output shapes
dummy_input = torch.randn(1, 3, 32, 32)
# This line will be removed to prevent creating an unused variable warning
_ = dummy_input.shape[1]

# Target maximum parameter count
target_params = 500000

# Design criteria
learning_rate = 0.01
momentum = 0.9

# Assign the dummy input shape to in_shape and out_shape
in_shape = dummy_input.shape
out_shape = (10,)  # Dummy output shape for classification

# Assign the dummy input shape to prm and device for consistency
prm = {'lr': learning_rate, 'momentum': momentum}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The actual model will be created below with the given constraints and resource limits