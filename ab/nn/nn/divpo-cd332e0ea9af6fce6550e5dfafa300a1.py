import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Conv2d(128, 4*128, kernel_size=1, stride=1),
            nn.BatchNorm2d(4*128),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*128, self.num_classes, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self._is_tracing = False
        self.to(device)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.bottleneck(x)
        x = self.classifier(x)
        return x.flatten(1)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

# Size and parameter constraint (must be <= 500000):
# Total params in Net should not exceed 500000, including all layers and parameters derived from hyperparameters.
def validate_size(m: torch.nn.Module, current: int, max_v: int) -> bool:
    if m._is_tracing:
        return True
    if 'num_batches_tracked' in m._parameters:
        m._parameters.pop('num_batches_tracked')
    if m.__class__.__name__ == 'BatchNorm2d':
        return validate_size(m, current + m.num_parameters(), max_v)
    return current + m.num_parameters() <= max_v

# Block for consideration (if it fits size constraint):
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super().__init__()
        self.identity = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expansion
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return self.relu(self.bn2(self.conv2(x))) + x
        out = self.relu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(out)) + x

# Resource check (must hold for valid model):
def validate_size_limit(model: torch.nn.Module, size_limit: int) -> bool:
    model._is_tracing = False
    model.eval()
    x = torch.empty(1, *model.in_shape[1:], device='cuda')
    r = torch.jit.trace(model, x, check_trace=False)
    return validate_size(r, 0, size_limit)