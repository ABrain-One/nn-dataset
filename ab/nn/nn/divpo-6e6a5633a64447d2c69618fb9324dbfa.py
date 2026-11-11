import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        return nn.ReLU(inplace=True)(out)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32, stride=1),
            Block(32, 64, stride=2),
            Block(64, 64, stride=1),
            Block(64, 128, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
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

    def __next_hyperparameter__(self, prm):
        values = {
            'lr': [0.01, 0.001, 0.0001],
            'momentum': [0.9, 0.8, 0.7]
        }
        for hp in prm:
            if hp in values:
                prm[hp] = values[hp][0]
                values[hp] = values[hp][1:]
        return prm

    def __next_param_value__(self, current, values):
        if current in values:
            return values[current][0]
        return None

    def apply_hyperparameters(self, prm):
        self.lr = self.__next_param_value__('lr', prm)
        self.momentum = self.__next_param_value__('momentum', prm)
        self.train_setup(prm)

    def __estimate_params__(self, x):
        x = x.to(self.device)
        self.eval()
        r = self.forward(x)
        return r.nelement()

    def __calculate_complexity__(self, x):
        total_ops = 0
        x = x.to(self.device)
        hook_handle = self.register_forward_hook(self.__count_ops__)
        self.forward(x)
        hook_handle.remove()
        return total_ops

    def __count_ops__(self, m, i, o):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            k = m.kernel_size[0] * m.kernel_size[1] * m.in_channels * m.out_channels
            flops = k * m.in_channels * m.out_channels / m.groups
            self.__total_ops__ += flops

    def __total_ops__(self):
        total_ops = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                k = m.kernel_size[0] * m.kernel_size[1] * m.in_channels * m.out_channels
                flops = k * m.in_channels * m.out_channels / m.groups
                total_ops += flops
        return total_ops

    def estimate_complexity(self, x):
        return self.__calculate_complexity__(x)

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['stem']
        del d['classifier']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        in_channels = 3
        num_classes = int(self.out_shape[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32, stride=1),
            Block(32, 64, stride=2),
            Block(64, 64, stride=1),
            Block(64, 128, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)