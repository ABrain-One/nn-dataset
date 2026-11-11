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
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W) — DO NOT hardcode
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 32)
        self.block3 = Block(32, 64)
        self.block4 = Block(64, 64)
        self.block5 = Block(64, 64)
        self.block6 = Block(64, 64)
        self.block7 = Block(64, 64)
        self.block8 = Block(64, 64)
        self.pool = nn.MaxPool2d(2)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
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
        x = self.block7(x)
        x = self.block8(x)
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

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}

    def __repr__(self):
        return super().__repr__() + f'(blocks: {self.block1} {self.block2} {self.block3} {self.block4} {self.block5} {self.block6} {self.block7} {self.block8})'

# helper for v2 (skip: 0.001599492337650645, 0.002643828198468979, 0.0014366443443001447, 0.0006781619176495971, 0.0003731736507517288)
# 0.001599492337650645 * 2.356734917659468 = 0.0036393549674594247
# 0.002643828198468979 * 2.356734917659468 = 0.006259331294887232
# 0.0014366443443001447 * 2.356734917659468 = 0.003400361557292387
# 0.0006781619176495971 * 2.356734917659468 = 0.0015757452256953506
# 0.0003731736507517288 * 2.356734917659468 = 0.0008802818278346844
# 0.0036393549674594247 + 0.006259331294887232 + 0.003400361557292387 + 0.0015757452256953506 + 0.0008802818278346844 = 0.01502446485547605
# 2.356734917659468 / 0.01502446485547605 = 156.33751319628536
# 156.33751319628536 * 0.002643828198468979 = 4.12067216362792
# 156.33751319628536 * 0.0014366443443001447 = 2.120183772508359
# 156.33751319628536 * 0.0006781619176495971 = 1.062164038350284
# 156.33751319628536 * 0.0003731736507517288 = 0.5698421102695048
# 4.12067216362792 + 2.120183772508359 + 1.062164038350284 + 0.5698421102695048 = 8.87295208495521
# 0.01502446485547605 / 8.87295208495521 = 0.0171126927