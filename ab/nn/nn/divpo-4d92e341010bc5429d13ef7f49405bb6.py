import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # pointwise
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            # depthwise
            nn.Conv2d(hidden_dim if expand_ratio != 1 else in_channels, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pointwise
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        layers = []
        # We will use inverted residuals for the feature extractor.
        # Here is a configuration that can be used as a starting point, but feel free to experiment with different block types and configurations.
        # Note that we will use the inverted residuals with expansion ratio 6 for the feature extractor.
        # The block should support stride=1 and stride=2, and we recommend using the default padding=1 for the convolutions.
        # The block should have the following arguments in the __init__ function:
        # self.stride = stride
        # assert stride in [1, 2]
        # Note that the block should use self.in_channels, self.out_channels, and self.exp_ratio
        # in place of hardcoded channel counts.
        # For example, the __init__ function of the inverted residual block could look like this:
        # def __init__(self, in_channels, out_channels, stride, exp_ratio):
        #     super(InvertedResidual, self).__init__()
        #     ...
        #     self.in_channels = in_channels
        #     self.out_channels = out_channels
        #     self.exp_ratio = exp_ratio
        #     ...
        # layers.append(InvertedResidual(in_channels, 32, stride=1, exp_ratio=6))
        # layers.append(InvertedResidual(32, 32, stride=2, exp_ratio=6))
        # layers.append(InvertedResidual(32, 32, stride=1, exp_ratio=6))
        # self.features = nn.Sequential(*layers)

        self.features = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.classifier = nn.Linear(32 * 6 * 6, num_classes)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data):
        self.train()
        for batch_idx, (data, target) in enumerate(train_data):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()