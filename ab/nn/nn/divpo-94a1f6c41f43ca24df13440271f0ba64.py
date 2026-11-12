import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.stem = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.block_1 = self._make_block(32, 32, 1, 3, stride=1)
        self.block_2 = self._make_block(32, 64, 1, 3, stride=2)
        self.block_3 = self._make_block(64, 64, 1, 3, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Linear(64, self.num_classes)

    def _make_block(self, in_channels, out_channels, block_num, block_strides, stride):
        blocks = []
        blocks.append(nn.BatchNorm2d(in_channels))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=block_strides,
                      stride=stride, bias=False, padding=1)
        )
        for _ in range(block_num - 1):
            blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=1,
                          stride=1, bias=False, padding=1)
            )
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.avg_pool(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
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