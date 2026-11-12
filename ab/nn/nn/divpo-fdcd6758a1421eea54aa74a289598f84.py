import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]   # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=1)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 32)
        self.block3 = Block(32, 64, kernel_size=3, padding=1, stride=2)
        self.block4 = Block(64, 64)
        self.block5 = Block(64, 64)
        self.block6_1 = Block(64, 128, kernel_size=3, padding=1, stride=2)
        self.block6_2 = Block(128, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6_1(x)
        x = self.block6_2(x)
        x = self.pool(x)
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
    _prev_length = 0

    def _append_block(self, name, cls, *args):
        '''Add block, auto-renumber layers '''
        assert self._prev_length <= len(self._modules)
        layers = {n: x for n, x in reversed(list(self._modules.items()))}
        prev_len = len(layers)
        for n, l in layers.items():
            if isinstance(l, Block):
                setattr(self, name + f'{prev_len+1}', cls(*args, **l.state_dict()))
                prev_len += 1
        assert self._prev_length <= len(self._modules)
        self._prev_length = len(self._modules)
        return prev_len

    def _downsample_block(self, channels, stride):
        return Block(channels, channels, stride=stride)

    _units = [
        Block,
        Block,
        Block,
        Block,
        _downsample_block,
        Block,
        Block,
        Block,
        _downsample_block,
        Block,
        Block,
    ]

    def build_cells(self, num_units, num_init_features, block_args, N=3, K=3, S=2, alpha=1.0):
        assert 1 <= S <= 2
        S = int(S * alpha)
        for _ in range(num_units):
            block_args.append((
                int(block_args[-1][0] * 1.5),  # channels
                min(int(block_args[-1][1] * 1.5), 2),  # repeat
                min(S, 2),  # stride
            ))
            if block_args[-1][1] == 2:
                self._append_block('unit', self._units[0], *block_args[-1])
            else:
                self._append_block('unit', self._units[1], *block_args[-1])
            self._append_block('unit', self._units[2], *block_args[-1])

        self._append_block('unit', self._units[3], *block_args[-1])

        for _ in range(N):
            self._append_block('unit', self._units[4], *block_args[-1])
            for _ in range(K):
                self._append_block('unit', self._units[5], *block_args[-1])
            self._append_block('unit', self._units[6], *block_args[-1])

        self._append_block('unit', self._units[7], *block_args[-1])

        # Downsample to 16
        for _ in range(S):
            self._append_block('unit', self._units[8], *block_args[-1])
            self._append_block('unit', self._units[9], *block_args[-1])
        self._append_block('unit', self._units[10], *block_args[-1])

        # Expand to 40
        for _ in range(int(3 * alpha) - 1):
            self._append_block('unit', self._units[0], *block_args[-1])
            self._append_block('unit', self._units[1], *block_args[-1])
            self._append_block('unit', self._units[2], *block_args[-1])

        self.block10 = self._units[3](**block_args[-1])
        self.block11 = self._units[4](**block_args[-1])
        self.block12 = self._units[5](**block_args[-1])

        self._modules.pop('unit3')
        self._modules.pop('unit4')
        self._modules.pop('unit5')

    def _block_params(self, channels, repeat, stride):
        return (channels, repeat, stride)

    def _block_units(self, N, K, S, block_args):
        block_args = [self._block_params(32, 1, 1)]
        self.build_cells(N, 32, block_args, N=N, K=K, S=S)
        self._append_block('unit1', self._units[3], *block_args[-1])
        self._append_block('unit2', self._units[4], *block_args[-1])
        self._append_block('unit3', self._units[5], *block_args[-1])