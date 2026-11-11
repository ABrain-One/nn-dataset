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
        self.module_0 = Block(32, 32)
        self.module_1 = Block(32, 32)
        self.module_2 = Block(32, 32)
        self.module_3 = Block(32, 32)
        self.module_4 = Block(32, 64)
        self.module_5 = Block(64, 64)
        self.module_6 = Block(64, 64)
        self.module_7 = Block(64, 64)
        self.module_8 = Block(64, 64)
        self.module_9 = Block(64, 64)
        self.module_10 = Block(64, 64)
        self.module_11 = Block(64, 64)
        self.module_12 = Block(64, 64)
        self.module_13 = Block(64, 64)
        self.module_14 = Block(64, 32)
        self.module_15 = Block(32, 32)
        self.module_16 = Block(32, 32)
        self.module_17 = Block(32, 32)
        self.module_18 = Block(32, 32)
        self.module_19 = Block(32, 32)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.module_0(x)
        x = self.module_1(x)
        x = self.module_2(x)
        x = self.module_3(x)
        x = self.module_4(x)
        x = self.module_5(x)
        x = self.module_6(x)
        x = self.module_7(x)
        x = self.module_8(x)
        x = self.module_9(x)
        x = self.module_10(x)
        x = self.module_11(x)
        x = self.module_12(x)
        x = self.module_13(x)
        x = self.module_14(x)
        x = self.module_15(x)
        x = self.module_16(x)
        x = self.module_17(x)
        x = self.module_18(x)
        x = self.module_19(x)
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(in_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(self.leakyrelu(self.batchnorm(x)))