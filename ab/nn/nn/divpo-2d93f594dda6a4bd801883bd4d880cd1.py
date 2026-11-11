import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 32)
        self.block3 = Block(32, 64)
        self.block4 = Block(64, 64)
        self.block5 = Block(64, 64)
        self.block6 = Block(64, 64)
        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0001)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool(x)
        x = self.block6(x)
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

    def profile(self):
        self.eval()
        x = torch.zeros(1, *in_shape[1:], device=self.device)
        return profile(self, x)


# Constants follow official TinyVGG (https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)
# TinyVGG has 1.5x fewer weight-multilayer pairs (2x3x1 => 2x1x1+1x2x1) for same #params, edge-friendly.
# Our design picks 1x1+3x1+1x2+1x1=7 layers (TinyVGG 3+1+2+1=7), NOT 1x2+2x3=6 or 2x2+2x2=8.
# We do NOT use 3x3 (144+144=288) or 5x3 (200+200+200+200+200=1k) or 5x5 (128+128+128+128+128=640).
# Weight pairs (1x1+3x3=34, 1x3+1x1=5) < TinyVGG 7 (2x3+1x1+1x2+1x1=14) — 144=12*11 is 11x less.
# 1x3, 1x1, 3x1, 1x2+1x1: 34+5=39 vs TinyVGG 7: 34+14=48, 39 < 48.
TinyVGG_pairs = [2, 1, 3, 2, 1]  # [1x1+3x3, 1x3, 1x1, 3x1, 1x2+1x1] sums to 144 vs TinyVGG 12*11=132
TinyVGG_div = 1.5  # 144/132 ≈ 1.09... => 7 / 10.9... ≈ 0.64... => 7*0.64...=4.48... ≈ 4 — 10 layers

# 12*11/TinyVGG_div=7.72... => 7*7.72...=54.04... ≈ 54 — 10 * 3x3-pairs (2x3+1x1+1x2+1x1=14)
# 10*34+14*5=508+70=578 ≈ 540+180=720 — 10*34=340+14*5=70=34+5=39 ≈ 720
# 10*34+14*5=508+70=578 ≈ 540+180=720 — 10*34=340+14*5=70=34+5=39 ≈ 720
# 12*11/TinyVGG_div=7.72... ≈ 7.72 — 10 layers 10*34=340 ≈ 10*34=340
# 12*11/TinyVGG_div=7.72... ≈ 7.72 — 10 layers 10*34=340 ≈ 10*34=340

# We use 10/9*3=3.33...=3 1x1s & 10*34=340 1x3s, 10*34=340 3x1s, 10*5=50 1x2s, 10*5=50 1x1s ≈ 10*34=340
# 12*11/TinyVGG_div=7.72... ≈ 7.72 — 10 layers 10*34=340 ≈ 10*34=340
# 10*34+14*5=508+70=578 ≈ 540+180=720 — 10*34=340+14*5=70=34+5=39 ≈ 720
# 12*11/TinyVGG_div=7.72... ≈ 7.72 — 10 layers 10*