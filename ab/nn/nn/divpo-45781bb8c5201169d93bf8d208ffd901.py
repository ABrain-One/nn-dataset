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
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 64)
        self.block3 = Block(64, 32)
        self.block4 = Block(32, 32)
        self.block5 = Block(32, 64)
        self.block6 = Block(64, 32)
        self.block7 = Block(32, 32)
        self.block8 = Block(32, 64)
        self.block9 = Block(64, 32)
        self.block10 = Block(32, 32)
        self.block11 = Block(32, 64)
        self.block12 = Block(64, 32)
        self.block13 = Block(32, 32)
        self.block14 = Block(32, 64)
        self.block15 = Block(64, 32)
        self.block16 = Block(32, 32)
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.pool(x)
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

# Optional: helper blocks
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def check_shape(prm):
    return 'in_shape' in prm and isinstance(prm['in_shape'], (list, tuple))

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def supported_hyperparameters():
    return {'lr', 'momentum'}

def prune_hyperparameters(prm, required_only):
    return {hp: prm[hp] for hp in required_only if hp in prm}

def create_initializer(prm):
    device = torch.device('cpu')  # non-edge friendly
    return Net(prm['in_shape'], prm['out_shape'], prm, device)

def train_setup(prm, net):
    lr = prm.get('lr', 0.01)
    momentum = prm.get('momentum', 0.9)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    return optimizer, criterion

def learn(data_loader, net, optimizer):
    net.train()
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
    return net