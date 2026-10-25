import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expand_ratio
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

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
        self.model_name = 'MobileNetV3_large_451_inverted_residual_block'
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Make a few helpers to avoid code duplication
        class Block(nn.Module):
            def __init__(self, in_channels, out_channels, stride, expand_ratio):
                super(Block, self).__init__()
                self.depthwise = InvertedResidual(in_channels, out_channels, stride, expand_ratio)

            def forward(self, x):
                return self.depthwise(x)

        class MobileNetV3_large_451_InvertedResidual_block(nn.Module):
            def __init__(self, in_channels, out_channels, stride, expand_ratio):
                super(MobileNetV3_large_451_InvertedResidual_block, self).__init__()
                self.block = Block(in_channels, out_channels, stride, expand_ratio)

            def forward(self, x):
                return self.block(x)

        self.features = nn.Sequential(
            MobileNetV3_large_451_InvertedResidual_block(32, 16, 1, 6),
            MobileNetV3_large_451_InvertedResidual_block(16, 24, 2, 6),
            MobileNetV3_large_451_InvertedResidual_block(24, 32, 1, 6),
            MobileNetV3_large_451_InvertedResidual_block(32, 64, 2, 6),
            MobileNetV3_large_451_InvertedResidual_block(64, 96, 1, 6),
            MobileNetV3_large_451_InvertedResidual_block(96, 160, 2, 6),
            MobileNetV3_large_451_InvertedResidual_block(160, 320, 1, 6),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(320, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.to(device)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'] if 'lr' in prm else 0.01,
            momentum=prm['momentum'] if 'momentum' in prm else 0.9,
            weight_decay=0.00001
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=prm.get('weight_decay', 0.00001)
        )

    def learn(self, train_data):
        self.train()
        for batch_idx, (data, target) in enumerate(train_data):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()