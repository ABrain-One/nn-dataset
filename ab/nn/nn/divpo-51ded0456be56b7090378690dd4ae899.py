import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # Block 1
        self.block1 = self._make_layer(32, 32, 3, True)

        # Block 2
        self.block2 = self._make_layer(32, 32, 3, True)

        # Block 3
        self.block3 = self._make_layer(32, 64, 3, True, True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

    def _make_layer(self, in_planes, planes, blocks, downsample=False, apply_dropout=False):
        layers = []
        if downsample:
            layers.append(Block(in_planes, planes, stride=2, apply_dropout=apply_dropout))
        else:
            layers.append(Block(in_planes, planes, apply_dropout=apply_dropout))
        for _ in range(blocks - 1):
            layers.append(Block(planes, planes, apply_dropout=apply_dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        # First block
        out = self.stem(x)
        out = self.relu(out)

        # Inverted residual blocks
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.pool(out)

        # Global average pooling and classifier
        out = self.avgpool(out)
        out = out.flatten(1)
        out = self.classifier(out)
        return out

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

# Simple block for the first part of the model
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, apply_dropout=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.dropout = nn.Dropout(0.3) if apply_dropout else nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(residual)
        out = self.relu(out)

        out = self.dropout(out)
        return out


def supported_hyperparameters():
    return {'lr', 'momentum'}


def check_bias_and_pad(m, i, o):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        assert m.bias is None
        assert m.padding == 1


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_size():
    model = Net(in_shape=(3, 32, 32), out_shape=(10,), prm={'lr': 0.01, 'momentum': 0.9}, device='cpu')
    onnx_exportable(model)
    assert count_params(model) <= 500000

def onnx_exportable(model):
    model.eval()
    for _ in range(2):  # multiple forward passes
        x = torch.rand(1, 3, 32, 32)
        out = model(x)
        assert out.shape[0] == 1
    # export and check for no errors
    model.cpu().detach()
    x = torch.rand(1, 3, 32, 32)
    torch.onnx.export(model, x, "temp.onnx", input_names=["input"], output_names=["output"])

def test_inference():
    model = Net(in_shape=(3, 32, 32), out_shape=(10,), prm={'lr': 0.01, 'momentum': 0.9}, device='cpu')
    model.eval()
    x = torch.rand(1, 3, 32, 32)
    out = model(x)
    assert out.shape[0] == 1

def test_backprop():
    model = Net(in_shape=(3, 32, 32), out_shape=(10,), prm={'lr': 0.01, 'momentum': 0.9}, device='cpu')
    model.train()
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    y = model(x)
    y.sum().backward()
    assert x.grad is not None