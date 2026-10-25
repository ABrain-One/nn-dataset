import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Net class with helper blocks (if needed)
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_conn = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True)
                )
            )

        layers.extend([
            nn.Conv2d(hidden_dim if expand_ratio != 1 else in_channels,
                      hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        layers.append(
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_conn:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.kernel_size = 3
        self.padding = 1
        self.stride = 1
        self.expand_ratio = 4
        self.channels = [32, 16, 32, 32, 32, 32, 32, 64, 64, 64]

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True),
            InvertedResidual(self.channels[0], self.channels[1], stride=1, expand_ratio=self.expand_ratio),
            InvertedResidual(self.channels[1], self.channels[2], stride=1, expand_ratio=self.expand_ratio),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(self.channels[2], self.channels[3], stride=1, expand_ratio=self.expand_ratio),
            InvertedResidual(self.channels[3], self.channels[4], stride=1, expand_ratio=self.expand_ratio),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(self.channels[4], self.channels[5], stride=1, expand_ratio=self.expand_ratio),
            InvertedResidual(self.channels[5], self.channels[6], stride=1, expand_ratio=self.expand_ratio),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(self.channels[6], self.channels[7], stride=1, expand_ratio=self.expand_ratio),
            InvertedResidual(self.channels[7], self.channels[8], stride=1, expand_ratio=self.expand_ratio),
            InvertedResidual(self.channels[8], self.channels[9], stride=1, expand_ratio=self.expand_ratio),
            nn.AdaptiveAvgPool2d((6, 6)),
        )

        self.classifier = nn.Linear(self.channels[9] * 6 * 6, num_classes)

    def __repr__(self):
        return 'model with ' + str(sum(p.numel() for p in self.parameters())) + ' params'

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.05),
            momentum=prm.get('momentum', 0.9),
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