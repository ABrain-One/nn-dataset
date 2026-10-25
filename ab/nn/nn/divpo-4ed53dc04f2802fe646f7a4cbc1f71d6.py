import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv1(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # derive from in_shape
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32),
            Block(32, 32)
        )
        self.layer1 = nn.Sequential(
            Block(32, 32, stride=1),
            Block(32, 32, stride=1)
        )
        self.layer2 = nn.Sequential(
            Block(32, 64, stride=2, padding=1),
            Block(64, 64, stride=1)
        )
        self.layer3 = nn.Sequential(
            Block(64, 128, stride=2, padding=1),
            Block(128, 128, stride=1)
        )
        self.layer4 = nn.Sequential(
            Block(128, 256, stride=2, padding=1),
            Block(256, 256, stride=1)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)
        self.to(device)

        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = x.to(self.device)
        x = self.layer2(x)
        x = x.to(self.device)
        x = self.layer3(x)
        x = x.to(self.device)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

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

# Dummy input to derive export shape (channels-first C x H x W)
dummy_input = torch.randn(1, 3, 32, 32)
model = Net(in_shape=[3, 32, 32], out_shape=[10], prm={'lr': 0.01, 'momentum': 0.9}, device='cpu')
model.train_setup({'lr': 0.01, 'momentum': 0.9})
output = model(dummy_input)

# Exported module with correct input and output layers
module = torch.nn.Module()
module.cpu()
module.eval()
module.to(device='cpu')
module.train = False
module.parameters = lambda: iter([])
module.state_dict = lambda: {'dummy'}
module.train_setup = lambda prm: None
module.forward = lambda x: model(x.to(device='cpu'))  # export only this forward()
module.class_name = 'Net'
module.unique_name = 'Net_dummy'
module.device = 'cpu'
module.id = 0
module.in_shape = [3, 32, 32]
module.out_shape = [10]
module.prm = {'lr': 0.01, 'momentum': 0.9}