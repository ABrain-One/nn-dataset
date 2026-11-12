import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return nn.functional.relu(self.layer(x))


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 32)
        self.block3 = Block(32, 64)
        self.block4 = Block(64, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.block5_1 = Block(64, 64)
        self.block5_2 = Block(64, 64)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.flat = nn.Flatten()
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

        self.to_device(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = self.build_optimizer(prm, self.parameters())

    def build_optimizer(self, prm, param_groups):
        return torch.optim.SGD(
            param_groups,
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def to_device(self, device):
        self.to(device)
        self.stem.to(device)
        self.block1.to(device)
        self.block2.to(device)
        self.block3.to(device)
        self.block4.to(device)
        self.block5_1.to(device)
        self.block5_2.to(device)
        self.avg_pool.to(device)
        self.flat.to(device)
        self.classifier.to(device)

    def __repr__(self):
        return f'Net({self.block1_name} + Double{self.block2_name} + Double{self.block3_name} + Double{self.block4_name} + Double{self.block5_name} (6P))'

    def __str__(self):
        return self.__repr__()

    block1_name = 'Block1'
    block2_name = 'Block2'
    block3_name = 'Block3'
    block4_name = 'Block4'
    block5_name = 'Block5'

    def __train_blocks__(self, mode=True):
        self.block1.train(mode)
        self.block2.train(mode)
        self.block3.train(mode)
        self.block4.train(mode)
        self.block5_1.train(mode)
        self.block5_2.train(mode)

    def train_setup(self, prm):
        self.to_device(self.device)
        self.__train_blocks__(True)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = self.build_optimizer(prm, self.parameters())

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5_1(x)
        x = self.block5_2(x)
        x = self.avg_pool(x)
        x = self.flat(x)
        return self.classifier(x)

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