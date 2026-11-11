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
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convs = nn.Sequential()
        self.channels = 32
        self.depth = 1
        self.block_depth = 1
        self.inc = 32
        self.strides = [1]
        self.feature_size = 32
        self.build_convs()
        self.drop_rate = 0.0
        self.dropout = nn.Dropout(p=self.drop_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.net = nn.Linear(self.channels, self.num_classes)

    def build_convs(self):
        self.convs = nn.Sequential()
        for d in range(self.depth):
            block = self.build_block(d)
            self.convs.add_module(f"block{d}", block)

    def build_block(self, d):
        block = nn.Sequential()
        for i in range(self.block_depth):
            block.add_module(
                f"conv{i}",
                nn.Conv2d(
                    self.channels,
                    self.channels,
                    kernel_size=3,
                    padding=1,
                    stride=self.strides[d],
                ),
            )
            block.add_module(f"relu{i}", nn.ReLU())
            block.add_module(f"bn{i}", nn.BatchNorm2d(self.channels))
            if i != self.block_depth - 1:
                self.drop_rate = 0.5
                self.dropout = nn.Dropout(p=self.drop_rate)
        return block

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.convs(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()