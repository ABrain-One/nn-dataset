import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        if len(in_shape) == 4:  # channels-first (C, H, W)
            in_channels = in_shape[1]
        else:
            in_channels = in_shape[0]
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.block1 = Block(32, 32, True)
        self.block2 = Block(32, 64, True)
        self.block3 = Block(64, 64, True)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
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
            weight_decay=prm.get('weight_decay', 0.0005)
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


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if downsample:
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
            self.downsample_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            identity = self.downsample_bn(self.downsample_conv(x))
        out += identity
        out = self.relu(out)
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

_ = Block(32, 32, False)  # dummy block for shape analysis
in_shape = (3, 32, 32)  # dummy channels-first shape, NOT 32*32*3
in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
assert in_channels == 3, "dummy shape should match input channels"

prm = {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0005}
device = torch.device('cuda')

model_bound_params = 500000
scopes = [[], ['block1'], ['block2'], ['block1', 'block2'], ['block3'], ['block1', 'block3'], ['block2', 'block3'], ['block1', 'block2', 'block3']]
for scope in scopes:
    model = Net(in_shape, out_shape=(10,), prm=prm, device=device)
    for name, child in model.named_children():
        if name not in scope:
            child.train = lambda _: None  # detach from train setup
    prev_abs = None
    while count_params(model) > model_bound_params:
        for child_name, child_module in reversed(list(model.named_children())):
            if count_params(child_module) > prev_abs:
                model._modules.pop(child_name)
                prev_abs = None
                break
        else:  # no child removed — prune this module
            prev_abs = count_params(model)
            model.avgpool = nn.AdaptiveAvgPool2d((4, 4))
    assert count_params(model) <= model_bound_params, f"did not prune enough (got {count_params(model)} > {model_bound_params})"
    _ = model(torch.randn(1, *in_shape))[0]  # execute to find the first error
    print(f"Selected scope: {scope}")