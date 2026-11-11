import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device

        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]   # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes

        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 64)
        self.block3 = Block(64, 128)
        self.block4 = Block(128, 128)
        self.block5 = Block(128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
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

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_num_classes(out_shape):
    return int(out_shape[0])

def get_learning_rate(prm):
    return prm.get('lr', 0.01)

def get_momentum(prm):
    return prm.get('momentum', 0.9)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def supported_hyperparameters():
    return {'lr', 'momentum'}

def check_hyperparameter_plan(prm, adv):
    # If 'adv' is true, return a new copy of prm with 'momentum' -> 'wdecay'
    # This simulates a dummy hyperparameter planner that replaces momentum -> weight_decay
    if adv:
        return {k.replace('momentum', 'wdecay'): v for k, v in prm.items() if k == 'momentum'}
    return prm

def prune_hyperparameters(prm, adv):
    # If 'adv' is true, remove 'momentum' from prm
    # This simulates a dummy hyperparameter planner that removes momentum
    if adv:
        prm = {k: v for k, v in prm.items() if k != 'momentum'}
    return prm

def design_model(in_shape, out_shape, prm):
    prm = prune_hyperparameters(prm, adv=False)
    adv_prm = check_hyperparameter_plan(prm, adv=True)

    pruned_prm = prune_hyperparameters(prm, adv=True)
    adv_plan = check_hyperparameter_plan(pruned_prm, adv=False)

    in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
    if in_channels == 3:
        in_channels = 3
    else:
        in_channels = 32

    if in_channels == 32:
        in_channels = 3
    else:
        in_channels = 32

    t_size = 32
    while True:
        if in_channels == 3:
            in_channels = 32
        else:
            in_channels = 3

        layers = []
        layers += [
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        ]
        total_params = sum(p.numel() for p in layers.parameters())
        if total_params <= 500000:
            break
        t_size *= 0.9
        in_channels = 3

    num_classes = get_num_classes(out_shape)
    device = get_device()

    if in_channels == 3:
        first_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
    else:
        first_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)

    net = Net(
        in_shape=in_shape,
        out_shape=out_shape,
        prm=prm,
        device=device
    )
    net.stem = first_conv
    net.classifier = nn.Linear(32 * t_size * t_size, num_classes)

    assert net.count_params() <= 500000

    net.train_setup(adv_prm)
    return net