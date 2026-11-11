import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class MultiSkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_blocks=3, kernel_size=3, padding=1):
        super().__init__()
        self.multi_skip = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            ) for _ in range(num_blocks)
        ])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = self._downsample_block(in_channels, out_channels, stride)
        self.kernel_size = kernel_size
        self.stride = stride

    def _downsample_block(self, in_channels, out_channels, stride):
        if stride != 1 or in_channels != out_channels:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        return nn.Sequential()

    def forward(self, x):
        residual = x
        multi_outputs = []
        for layer in self.multi_skip:
            multi_outputs.append(layer(x))
        x = torch.stack(multi_outputs, dim=-1)
        x = torch.mean(x, dim=-1)
        x = self.relu(x + self.downsample(residual))
        return x

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = self._downsample_block(in_channels, out_channels, stride)

    def _downsample_block(self, in_channels, out_channels, stride):
        if stride != 1 or in_channels != out_channels:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        return nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.attention(x)
        x = self.relu(x + self.downsample(residual))
        return x

class Block(nn.Module):
    """ Multi-Skip with multi-scale attention """
    def __init__(self, in_channels, out_channels, stride=1, num_blocks=3, kernel_size=3, padding=1):
        super().__init__()
        self.multi_skip = MultiSkipBlock(in_channels, out_channels, stride, num_blocks, kernel_size, padding)
        self.attention = AttentionBlock(in_channels, out_channels, stride, kernel_size, padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.multi_skip(x)
        x = self.attention(x)
        return self.pool(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Linear(64 * 6 * 6, out_shape[0])
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.features(x)
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