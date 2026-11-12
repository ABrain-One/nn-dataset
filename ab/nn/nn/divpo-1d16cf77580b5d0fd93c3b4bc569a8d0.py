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
        self.block11 = Block(128, 128, True)
        self.block12 = Block(128, 64, True)
        self.block13 = Block(64, 64, True)
        self.block14 = Block(64, 32, True)
        self.block15_1 = Block(32, 32, True)
        self.block15_2 = Block(32, 32, True)
        self._avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(32 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.block5(x)
        x = self.block6_1(x)
        x = self.block6_2(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.block15_1(x)
        x = self.block15_2(x)
        x = self._avg_pool(x)
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

# Block template (add as many as you like in the Net initializer):
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not batchnorm))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(True))
        self.unit = nn.Sequential(*layers)

    def forward(self, x):
        return self.unit(x)


# Utility function to compute the number of channels after block sequences
def channels_after(block_counts, in_channels):
    out_channels = in_channels
    for count in block_counts:
        for _ in range(count):
            out_channels = channels_after_one(out_channels)
    return out_channels

# Per-block channels change (32-128 repeats, 128-64 repeats, 128-32 repeats, 32-32 then 32-64);
# add a helper to avoid code duplication:
def channels_after_one(in_channels):
    # 32-128 repeats 2x, 128-64 repeats 2x, 128-32 repeats 2x, 32-32 then 32-64
    # 1st round: 32->128, 128-128, 128-64, 64-64
    # 2nd round: 64->128, 128-128, 128-32, 32-32
    # 3rd round: 32->128, 128-128, 128-64, 64-64
    # 1st-3rd -> 128-128-64-64
    # 1st 32-64 -> 32
    # 2nd 32-64 -> 64
    # 3rd 32-64 -> 64
    # -> 128-128-64-64-32-64
    # 64, 128, 128, 128, 32, 64
    # 64-128-128-128-32-64
    # 128, 128, 128, 32, 64
    # 128-128-64-64
    # 128, 64
    # 128-64
    # 64
    # 32-128-128-64-64
    # 32-128-128-64-64
    # 32-64
    # 64
    # 128, 128, 64, 64
    # 128-128-64-64
    # 128, 64
    # 128-64
    # 64
    if in_channels == 32:
        return 128
    if in_channels == 128:
        return 64
    assert in_channels == 64
    return 32

# Due to the block patterns above, 32, 64, 1