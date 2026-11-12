import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, padding=1),
            Block(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self._create_layers(32, num_classes)
        self.classifier = nn.Linear(32, num_classes)

    def _create_layers(self, in_channels, num_classes):
        layers = []
        # Replace with your own layer design
        layers.append(Block(in_channels, 32, kernel_size=3, padding=1))
        # Add more Block layers and transition blocks as needed
        # ...
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = x.mean(dim=(2, 3))  # Global average pooling
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

    def _count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def check_shape(shape):
    assert len(shape) == 4
    assert shape[1] in {1, 3}
    assert all(200 <= s <= 400 for s in shape[2:])

def make_it_fast(model):
    # Ensure we're testing a valid path
    if model.stem is model.features: return model
    for child in model.children():
        if isinstance(child, (Block, Transition)): make_it_fast(child); return
    assert False  # no matching path
    return None

def prune_to_size(model, total_params):
    model = model.to_legacy_segment()
    utils.prune_to_size(model, total_params)
    return Net(in_shape, out_shape, prm, device).features

def _make_block(in_channels, out_channels, block_count):
    layers = []
    for i in range(block_count):
        layers.append(Block(in_channels if i == 0 else out_channels, out_channels))
    return Block(in_channels, out_channels, _make_block(out_channels, out_channels, block_count - 1) if block_count > 1 else None)

class Transition(Block):
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__(in_channels, out_channels, kernel_size=1, stride=1 if pool else 2, padding=0, dilation=1, groups=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x):
        x = super().forward(x)
        return self.pool(x)

def _make_layers(in_channels, num_classes):
    downsample = Transition(in_channels, 32, pool=False)
    layers = [downsample]
    # Replace with your own layer design
    layers.append(_make_block(32, 32, 2))
    # Add more Transition blocks and make_block instances as needed
    # ...
    return layers

def __repr__(self):
    return f'{self.__class__.__name__}(in_channels={self.in_channels} -> {self.out_channels})'

def supported_hyperparameters():
    return {'lr', 'momentum'}

def check_shape(shape):
    assert len(shape) == 4
    assert shape[1] in {1, 3}
    assert all(200 <= s <= 400 for s in shape[2:])

def prune_to_size(model, total_params):
    model = model.to_legacy_segment()
    utils.prune_to_size(model, total_params)
    return Net(in_shape, out_shape, prm, device).features

def _make_block(in_channels, out_channels, block_count):
    layers = []
    for i in range(block_count):
        layers.append(Block(in_channels if i == 0 else out_channels, out_channels))
    return Block(in_channels, out_channels, _make_block(out_channels, out_channels, block_count - 1) if block_count > 1 else None)

def _create_layers(self, in_channels, num_classes):
    downsample = Transition(in_channels, 32, pool=False)
    layers = [downsample]
    # Replace with your own layer design
    layers.append(_make_block(32, 32, 2))
    # Add more Transition blocks and make_block instances as needed
    # ...
    self.features = nn.Sequential(*layers)

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

def make_it_fast(model):
    # Ensure we're testing a valid path
    if model.stem is model.features: return model
    for child in model.children():
        if isinstance(child, (Block, Transition)): make_it_fast(child); return
    assert False  # no matching path
    return None

def __init__(self, in_shape, out_shape, prm, device):
    self.device = device
    in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
    num_classes = out_shape[0]
    self.stem = nn.Sequential(
        Block(in_channels, 32, kernel_size=3, padding=1),
        Block(32, 32, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self._create_layers(32, num_classes)
    self.classifier = nn.Linear(32, num_classes)

def forward(self, x):
    x = self.stem(x)
    x = self.features