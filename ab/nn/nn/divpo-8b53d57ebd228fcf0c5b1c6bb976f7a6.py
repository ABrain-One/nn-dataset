import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.build_layer(32, 32, 2, first_pool=False)
        self.layer2 = self.build_layer(32, 64, 2)
        self.layer3 = self.build_layer(64, 128, 2)
        self.layer4 = self.build_resblock(128, 128, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.0)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self._is_fwd_comp = False

    def build_layer(self, in_channels, out_channels, stride, first_pool=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if first_pool:
            layers.insert(0, nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def build_resblock(self, in_channels, out_channels, repeat):
        layers = []
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(repeat - 1):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def compile(self, prm):
        self._is_fwd_comp = True
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def train_forward(self, batch):
        if self._is_fwd_comp:
            return self.forward(batch)
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        return self.learn((inputs, labels))

    def export(self):
        assert self._is_fwd_comp, 'Compile model first!'
        return nn.Sequential(*list(self.children())[:-1])

def supported_hyperparameters():
    return {'lr', 'momentum'}

def check_shapes_prm(prm):
    if 'lr' in prm:
        assert 0 < prm['lr'] < 1, 'Invalid learning rate'
    if 'momentum' in prm:
        assert 0 < prm['momentum'] <= 1, 'Invalid momentum'

def validate_prm(prm):
    check_shapes_prm(prm)
    return {k: v for k, v in prm.items() if k in supported_hyperparameters()}

def get_learning_rate(prm):
    return prm.get('lr', 0.01)

def get_momentum(prm):
    return prm.get('momentum', 0.9)

def get_dropout(prm):
    return prm.get('dropout', 0.0)

def get_weight_decay(prm):
    return prm.get('weight_decay', 0.0)

def get_out_channels(channels):
    return sum(int(c) for c in channels.split(','))

def get_conv_groups(out_channels):
    return out_channels // 16

def get_kernel_size(groups):
    return 3 if groups > 1 else 1

def get_num_filters(channels, kernel_size):
    return int(channels * kernel_size / 2)

def get_padding(num_filters, kernel_size):
    return (kernel_size - 1) // 2

def get_num_blocks(num_filters, num_classes):
    return int(num_classes / num_filters) + 1

def get_in_channels(channels):
    return sum(int(c) for c in channels.split(','))

def get_hyperparameters(channels, num_classes, prm):
    prm = validate_prm(prm)
    groups = get_conv_groups(get_out_channels(channels))
    kernel_size = get_kernel_size(groups)
    num_filters = get_num_filters(get_out_channels(channels), kernel_size)
    padding = get_padding(num_filters, kernel_size)
    num_blocks = get_num_blocks(num_filters, num_classes)
    in_channels = get_in_channels(channels)
    return {
        'lr': get_learning_rate(prm),
        'momentum': get_momentum(prm),
        'dropout': get_dropout(prm),
        'weight_decay': get_weight_decay(prm),
        'num_blocks': num_blocks,
        'num_filters': num_filters,
        'in_channels': in_channels,
        'kernel_size': kernel_size,
        'padding': padding,
        'out_channels': get_out_channels(channels)
    }

def supported_hyperparameters():
    return {'lr', 'momentum'}

def is_channel_last(shape):
    return shape[-1] != 1

def is_odd(n):
    return n % 2 == 1

def get_out_channels(channels):
    return sum(int(c) for c in channels.split(','))

def get_kernel_size(num_layers):
    return 3 if num_layers % 3 == 1 else 1

def get_num_filters(channels, kernel_size):
    return int(channels * kernel_size)

def get_hyperparameters(channels, num_classes, prm):
    prm