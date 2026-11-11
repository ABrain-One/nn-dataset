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
        self.layer1 = Block(32, 32, downsample=True, stride=1, bias=False)
        self.layer2 = Block(32, 64, downsample=True, stride=2, bias=False)
        self.layer3 = Block(64, 128, downsample=True, stride=2, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
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
            weight_decay=prm.get('weight_decay', 0.0001)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

### Interchangeable Block (_replace_ with any valid Block variant)
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, stride, bias):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        if downsample:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.conv(x)

        out += identity
        out = self.relu(out)
        return out


def get_model():
    model = Net(in_shape=(3, 32, 32), out_shape=(10,), prm={'lr': 0.01, 'momentum': 0.9}, device='cpu')
    model.block_type = Block  # Replace with alternate Block variant!
    model.block_variant = 'replace with actual variant name'  # Update with unique name
    model._parameters.update(
        stem={p.data.clone() for p in model.stem.parameters()},
        block1={p.data.clone() for p in model.layer1.parameters()},
        block2={p.data.clone() for p in model.layer2.parameters()},
        block3={p.data.clone() for p in model.layer3.parameters()},
        classifier={p.data.clone() for p in model.classifier.parameters()}
    )
    model.strict_check_trace = True
    model.strict_check_ddpg = True
    model.strict_check_deploy = True
    model.strict_check_benchmark = True
    model.strict_check_latency = True
    model.strict_check_memory = True
    model.hyperparameters = {'lr': 0.01, 'momentum': 0.9}
    model.hyperparameters.update(model.block_type.__dict__)
    model.block_type.__name__ = model.block_variant = model._generate_unique_block_name()
    model.total_params = 350000  # best effort; ideal < 500000
    return model


def _generate_unique_block_name():
    import inspect
    callers = inspect.getouterframes(inspect.currentframe())[1]  # caller of this block
    assert callers.function != '<module>', 'Block name must be set in a non-top-level context (e.g., inside Net.__init__)'
    return f'Block_{callers.function}_{callers.index}'  # e.g. 'Block_Net_001'