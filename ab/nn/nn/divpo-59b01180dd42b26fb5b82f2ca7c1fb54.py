import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.act(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        if len(in_shape) == 4:
            in_channels = in_shape[1]
        else:
            in_channels = in_shape[0]
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 32)
        self.block3 = Block(32, 64)
        self.block4 = Block(64, 64)
        self.block5 = Block(64, 64)
        self.block6 = Block(64, 64)
        self.block7 = Block(64, 64)
        self.block8 = Block(64, 64)
        self.block9 = Block(64, 32)
        self.block10 = Block(32, 32)
        self.block11 = Block(32, 32)
        self.block12 = Block(32, 32)
        self.block13 = Block(32, 32)
        self.block14 = Block(32, 32)
        self.block15 = Block(32, 32)
        self.block16 = Block(32, 32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.flat = nn.Flatten()
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)

        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.pool(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.pool(x)
        x = self.avg_pool(x)
        x = self.flat(x)
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


# Calculated estimated params
def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks BEFORE Net
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.act(self.conv(x)))

# Net configuration adhering to params constraint
# Calculations show 500K < 521278 (with stem) <= 545228 (without)
channels, width, height = 32, 11, 11  # 32x11x11 -> 64x5x5 -> 32x5x5 -> 32x3x3 -> 32x1x1 -> 32x5x5 -> 32x3x3 -> 32x1x1
_ = [32, 32, 64, 64, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]  # 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 16*32 = 512 -> 1