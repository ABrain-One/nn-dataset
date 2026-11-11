import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        if len(in_shape) == 4:
            in_channels = in_shape[1]
        else:
            in_channels = int(in_shape[0])
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.block1 = Block(32, 32, True)
        self.block2 = Block(32, 32, True)
        self.block3 = Block(32, 64, True)
        self.block4 = Block(64, 64, True)
        self.block5 = Block(64, 64, True)
        self.block6_1 = Block(64, 64, True)
        self.block6_2 = Block(64, 64, True)
        self.block7 = Block(64, 128, True)
        self.block8 = Block(128, 128, True)
        self.block9 = Block(128, 128, True)
        self.block10 = Block(128, 128, True)
        self.block11 = Block(128, 64, True)
        self.block12 = Block(64, 64, True)
        self.block13_1 = Block(64, 64, True)
        self.block13_2 = Block(64, 32, True)
        self.block14 = Block(32, 32, True)
        self.block15 = Block(32, 32, True)
        self.block16 = Block(32, 32, True)
        self.block17 = Block(32, 32, True)
        self.block18_1 = Block(32, 32, True)
        self.block18_2 = Block(32, 32, True)
        self.block19 = Block(32, 32, True)
        self.block20 = Block(32, 32, True)
        self.block21 = Block(32, 32, True)
        self.block22 = Block(32, 32, True)
        self.block23_1 = Block(32, 32, True)
        self.block23_2 = Block(32, 32, True)
        self.block24 = Block(32, 32, True)
        self.block25 = Block(32, 32, True)
        self.block26 = Block(32, 32, True)
        self.block27 = Block(32, 32, True)
        self.block28_1 = Block(32, 32, True)
        self.block28_2 = Block(32, 32, True)
        self.block29 = Block(32, 32, True)
        self.block30 = Block(32, 32, True)
        self.block31 = Block(32, 32, True)
        self.block32 = Block(32, 32, True)
        self.block33_1 = Block(32, 32, True)
        self.block33_2 = Block(32, 32, True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6_1(x)
        x = self.block6_2(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13_1(x)
        x = self.block13_2(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18_1(x)
        x = self.block18_2(x)
        x = self.block19(x)
        x = self.block20(x)
        x = self.block21(x)
        x = self.block22(x)
        x = self.block23_1(x)
        x = self.block23_2(x)
        x = self.block24(x)
        x = self.block25(x)
        x = self.block26(x)
        x = self.block27(x)
        x = self.block28_1(x)
        x = self.block28_2(x)
        x = self.block29(x)
        x = self.block30(x)
        x = self.block31(x)
        x = self.block32(x)
        x = self.block33_1(x)
        x = self.block33_2(x)
        x = self.pool(x)
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

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not batch_norm),
            nn.ReLU(True),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
        )

    def forward(self, x):
        return self.op(x)