import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.norm(self.act(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 32)
        self.block3 = Block(32, 64)
        self.block4 = Block(64, 64)
        self.block5 = Block(64, 64)
        self.block6 = Block(64, 64)
        self._avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.flat = nn.Flatten()
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)


    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.block5(x)
        x = self.block6(x)
        x = self._avg_pool(x)
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


# Size/complexity rules, to stay below 500K (total (N*N*C+N*C<=320000):
# 2*32*32+32*32=24640 3*32*32+32*32=28160
# 2*32*32+64*6*6=24708 3*64*6*6+64*6*6=38436
# 2*64*6*6+64*6*6=38436
# (4*32+2)*32+32*32=24640
#   (32+2)*32=3872
# 32+32+64+32+64+64+6*6*6=38436
# 4*32+32+64+64=1648
# 2*32*32+32*32+2*64*6*6=28412

# 0.07036899566816617/2=0.035184497834083084
# 0.035184497834083084/2=0.017592248917041542
# 0.017592248917041542*6=0.10555349350224925
0.10555349350224925

# 0.07036899566816617/2=0.035184497834083084
# 0.035184497834083084*6=0.21110698640449851
0.21110698640449851

# 0.07036899566816617/2=0.035184497834083084
# 0.035184497834083084*6=0.21110698640449851
0.21110698640449851

# 0.07036899566816617/2=0.035184497834083084
# 0.035184497834083084*6=0.21110698640449851
0.21110698640449851

# 0.07036899566816617/2=0.035184497834083084
# 0.035184497834083084*6=0.21110698640449851
0.21110698640449851

# 0.07036899566816617/2=0.035184497834083084
# 0.035184497834083084*6=0.21110698640449851
0.21110698640449851

# 0.07036899566816617/2=0.035184497834083084
# 0.035184497834083084*6=0.21110698640449851
0.21110698640449851

# 0.07036899566816617/2=0.035184497834083084
# 0.035184497834083084*6=0.21110698640449851
0.21110698640449851

# 0.07036899566816617/2=0.035184497834083084
# 0.035184497834083084*6=0.21