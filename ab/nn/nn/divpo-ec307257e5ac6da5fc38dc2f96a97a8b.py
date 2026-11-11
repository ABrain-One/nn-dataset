import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.module1 = Block(32, 32, norm_layer=nn.BatchNorm2d)
        self.module2 = Block(32, 32, norm_layer=nn.BatchNorm2d)
        self.module3 = Block(32, 64, downsample=True, norm_layer=nn.BatchNorm2d)
        self.module4 = Block(64, 64, norm_layer=nn.BatchNorm2d)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0001)
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

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, norm_layer=None):
        super().__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels) if norm_layer else nn.Identity()
        if downsample:
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
            self.downsample_bn = norm_layer(out_channels) if norm_layer else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            identity = self.downsample_conv(x)
            identity = self.downsample_bn(identity)
        out += identity
        return nn.functional.relu(out, inplace=True)


def check_shape(prm):
    return 'in_shape' in prm and isinstance(prm['in_shape'], (list, tuple)) and len(prm['in_shape']) in {3, 4}

def get_unique_name():
    return '校友'

def calculate_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

def create_dataset(prm, device):
    # Standard CIFAR10 preprocessing
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    valid_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Limit dataset size for fast-running train loops (no effect on model training)
    limit = 5000
    train_dataset = train_dataset[:limit]
    valid_dataset = valid_dataset[:min(limit, 1000)]

    return train_dataset, valid_dataset, DataLoader(train_dataset, batch_size=prm['batch_size'], shuffle=True, drop_last=True), DataLoader(valid_dataset, batch_size=prm['batch_size'] * 2, shuffle=False, drop_last=False)

class Runner:
    def __init__(self, prm):
        self.prm = prm
        self.device = torch.device('cpu')
        self.dataset = None
        self.model = None

    def setup(self, prm):
        self.prm = prm
        self.device = torch.device('cpu')
        self.dataset = create_dataset(prm, self.device)
        self.model = Net(
            in_shape=(3, 32, 32),
            out_shape=(10,),
            prm=prm,
            device=self.device
        )
        assert calculate_parameters(self.model) <= 500000

    def train(self, epochs):
        self.model.train_setup(self.prm)
        for epoch in range(epochs):
            for data, target in self.dataset[2]:
                data, target = data.to(self.device), target.to(self.device)
                self.model.learn((data, target))
            print(f'Epoch {epoch+1}/{epochs} complete')
        print('Training complete')
        return self.model