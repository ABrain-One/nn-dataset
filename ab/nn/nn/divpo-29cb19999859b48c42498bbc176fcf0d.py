import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion, se_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = int(in_channels * expansion)
        self.use_se = se_ratio is not None and se_ratio > 0

        if self.use_se:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels, se_channels, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(se_channels, in_channels, 1, bias=True),
                nn.Sigmoid()
            )
        else:
            self.se = nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv = nn.Identity()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.stride == 1 and self.conv is self.conv:
            residual = x
        else:
            residual = self.conv(x)
        x = self.se(x)
        x = self.block(x)
        x = x + residual
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.dropout = prm.get('dropout', 0.0)
        self.features = self.build_features()
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        layers += [
            InvertedResidual(32, 32, stride=1, expansion=1, se_ratio=0.25),
            InvertedResidual(32, 32, stride=1, expansion=6, se_ratio=0.25),
            InvertedResidual(32, 32, stride=1, expansion=6, se_ratio=0.25),
            InvertedResidual(32, 32, stride=1, expansion=6, se_ratio=0.25),
            InvertedResidual(32, 32, stride=1, expansion=6, se_ratio=0.25),
            InvertedResidual(32, 32, stride=1, expansion=6, se_ratio=0.25),
            InvertedResidual(32, 32, stride=1, expansion=6, se_ratio=0.25),
            InvertedResidual(32, 32, stride=1, expansion=6, se_ratio=0.25),
            InvertedResidual(32, 32, stride=1, expansion=6, se_ratio=0.25),
            InvertedResidual(32, 32, stride=1, expansion=6, se_ratio=0.25),
            InvertedResidual(32, 32, stride=1, expansion=6, se_ratio=0.25),
            InvertedResidual(32, 32, stride=1, expansion=6, se_ratio=0.25),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
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