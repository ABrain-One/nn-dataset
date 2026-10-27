import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block1 = Block(32, 32, prm, 1)
        self.block2 = Block(32, 64, prm, 2)
        self.block3 = Block(64, 128, prm, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

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

# Block (helper) is defined here
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, prm, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Sequential()
        
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.downsample(residual)
        out = self.relu(out)
        return out

# Generate a random 500000-ish parameter configuration
def generate_random_prm():
    prm = {'lr': 0.01, 'momentum': 0.9}
    # Ensure the total number of parameters is close to 500000
    layers = []
    prev_channels = 3
    for _ in range(5):
        out_channels = prev_channels * (2 if random.random() < 0.5 else 1)
        layers.append((f'block{len(layers) + 1}', Block(prev_channels, out_channels, prm, 1)))
        prev_channels = out_channels
    prm['blocks'] = layers
    # Ensure the total number of parameters is under 500000
    total_params = sum(p.numel() for layer, _ in layers for p in layer.parameters())
    if total_params > 500000:
        # Remove layers until we get under the limit
        while total_params > 500000:
            _, block = layers.pop(random.randint(0, len(layers) - 1))
            total_params -= sum(p.numel() for p in block.parameters())
    prm['blocks'] = layers
    return prm

def get_memory_format(channels_last):
    return torch.channels_last if channels_last else torch.contiguous_format

def supported_hyperparameters():
    return {'lr', 'momentum'}

def get_equivalent_block(block, channels_last):
    memory_format = get_memory_format(channels_last)
    return Block(
        in_channels=block.conv1.in_channels,
        out_channels=block.conv1.out_channels,
        prm=block.prm,
        stride=block.stride
    ).to(memory_format=memory_format)

def supported_hyperparameters():
    return {'lr', 'momentum'}

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

# The created model should not use external blocks defined outside this scope.
# All block definitions should be inside the Net class body.
# Block (name: Block) should have the following arguments: in_channels, out_channels, prm, stride.
# The created model should have an equivalent or slightly smaller parameter count than the one passed to generate_random_prm().