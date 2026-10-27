import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.use_skip = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expansion_factor
        self.conv = nn.Sequential(
            # pointwise
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # depthwise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pointwise
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_skip:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        dropout = prm.get('dropout', 0.2)
        # Layer 0: Gross Input Layer
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        # Layer 1: Inverted Residual Block (3x)
        self.layer1 = InvertedResidual(32, 32, 1, 3)
        self.layer2 = InvertedResidual(32, 32, 1, 3)
        self.layer3 = InvertedResidual(32, 32, 1, 3)
        # Layer 2: Transition Block
        self.layer2_1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        # Layer 3: Inverted Residual Block (2x)
        self.layer3_1 = InvertedResidual(32, 32, 1, 3)
        self.layer3_2 = InvertedResidual(32, 32, 1, 3)
        # Layer 4: Transition Block
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
        )
        # Layer 5: Inverted Residual Block (4x)
        self.layer5_1 = InvertedResidual(64, 64, 1, 3)
        self.layer5_2 = InvertedResidual(64, 64, 1, 3)
        self.layer5_3 = InvertedResidual(64, 64, 1, 3)
        self.layer5_4 = InvertedResidual(64, 64, 1, 3)
        # Final layer:
        self.conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2_1(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer4(x)
        x = self.layer5_1(x)
        x = self.layer5_2(x)
        x = self.layer5_3(x)
        x = self.layer5_4(x)
        x = self.conv(x)
        x = self.dropout(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return x.flatten(1)

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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()