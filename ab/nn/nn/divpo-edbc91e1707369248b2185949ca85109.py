import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def is_even(num):
    return num % 2 == 0

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.dropout_rate = prm.get('dropout', 0.2)

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.resblock1 = self.ResBlock(64, 64, stride=1, downsample=False)
        self.resblock2 = self.ResBlock(64, 128, stride=2, downsample=True)
        self.resblock3 = self.ResBlock(128, 128, stride=1, downsample=False)
        self.resblock4 = self.ResBlock(128, 256, stride=2, downsample=True)
        self.resblock5 = self.ResBlock(256, 256, stride=1, downsample=False)
        self.resblock6 = self.ResBlock(256, 256, stride=1, downsample=False)
        self.resblock7 = self.ResBlock(256, 512, stride=2, downsample=True)
        self.resblock8 = self.ResBlock(512, 512, stride=1, downsample=False)
        self.resblock9 = self.ResBlock(512, 512, stride=1, downsample=False)
        self.resblock10 = self.ResBlock(512, 512, stride=1, downsample=False)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.resblock9(x)
        x = self.resblock10(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data: torch.utils.data.DataLoader) -> None:
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    class ResBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int, downsample: bool) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.batchnorm1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.batchnorm2 = nn.BatchNorm2d(out_channels)

            if downsample:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.downsample = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            identity = self.downsample(x)

            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = self.relu(x)

            x = self.conv2(x)
            x = self.batchnorm2(x)

            x += identity
            x = self.relu(x)
            return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size(self) -> float:
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size

    def to_device(self, device: torch.device) -> None:
        self.to(device)

    def learn(self, train_data: torch.utils.data.DataLoader) -> None:
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()