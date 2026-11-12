import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        if len(in_shape) == 4:
            in_channels = in_shape[1]  # channels-first (C, H, W)
        else:
            in_channels = in_shape[0]
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.layer1 = Block(32, 32)
        self.layer2 = Block(32, 32)
        self.layer3 = Block(32, 64)
        self.layer4 = Block(64, 64)
        self.layer5 = Block(64, 64)
        self.layer6 = Block(64, 96)
        self.layer7 = Block(96, 96)
        self.layer8 = Block(96, 96)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(96 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.layer3(x)
        x = self.layer4(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.layer8(x)
        x = self.avgpool(x)
        x = x.permute(0, 2, 3, 1).contiguous()
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

# Optional: helper blocks AFTER Net class
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.norm(self.act(self.conv(x)))
    def fuse(self):
        if self.norm.affine:
            self.conv.weight.data.mul_(self.norm.weight.data.unsqueeze(-1).unsqueeze(-1))
            self.conv.bias.data.add_(self.norm.bias.data)
        self.norm.running_mean.zero_()
        self.norm.running_var.fill_(1)
        self.norm = nn.Identity()