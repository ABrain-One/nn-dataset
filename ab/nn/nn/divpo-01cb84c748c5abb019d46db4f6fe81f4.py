import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple[int, int, int, int], out_shape: tuple[int], prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.conv_block1 = self._make_conv_block(32, 32, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_block2 = self._make_conv_block(32, 64, 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(64 * 7 * 7, num_classes)

    def _make_conv_block(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Module:
        blocks = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        blocks += [nn.BatchNorm2d(out_channels)] + [nn.ReLU()] * 2
        for _ in range(1, num_blocks):
            blocks += [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)]
            blocks += [nn.BatchNorm2d(out_channels)] + [nn.ReLU()] * 2
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv_block1(x)
        x = self.maxpool1(x)
        x = self.conv_block2(x)
        x = self.maxpool2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data: torch.utils.data.DataLoader):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()