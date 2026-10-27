import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])  # number of target classes
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.5))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

                                     

    def get_flops(self, input_shape):
        input = torch.rand(input_shape, device=self.device)
        # count conv2d and linear
        flops = torchvision.models.get_model_complexity_info(self, input, device=self.device, verbose=False)[0]
        return flops

    def get_params(self):
        return sum(p.numel() for p in self.parameters())
                                     
                                             

    class SEBlock(nn.Module):
        def __init__(self, channel, ratio=16):
            super(SEBlock, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)

    class ResidualUnit(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, se_ratio=16):
            super(ResidualUnit, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.se = SEBlock(out_channels, se_ratio)
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.downsample = None

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.se(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu(out)
            return out

    class Hourglass(nn.Module):
        def __init__(self, in_channels, out_channels, depth, se_ratio=16):
            super(Hourglass, self).__init__()
            self.hg = nn.ModuleList([
                ResidualUnit(in_channels, out_channels, stride=1, se_ratio=se_ratio)
            ] + [
                ResidualUnit(out_channels, out_channels, stride=2, se_ratio=se_ratio)
            ] + [
                ResidualUnit(out_channels, out_channels, stride=1, se_ratio=se_ratio)
            ] * depth + [
                ResidualUnit(out_channels, out_channels, stride=1, se_ratio=se_ratio)
            ])
            self.res = ResidualUnit(in_channels, out_channels, stride=1, se_ratio=se_ratio)

        def forward(self, x):
            residual = x
            out = self.hg[0](x)
            for i in range(1, len(self.hg)):
                if i % 3 == 1:
                    residual = self.hg[i](residual)
                out = self.hg[i](out) + residual
            out = self.hg[-1](out)
            return out

    class HourglassNet(nn.Module):
        def __init__(self, in_shape, out_shape, prm, device):
            super(HourglassNet, self).__init__()
            self.device = device
            self.in_channels = in_shape[1]  # channels-first (C, H, W)
            num_classes = int(out_shape[0])  # number of target classes
            self.stem = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3)
            self.bn = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.hourglass = Hourglass(64, 128, depth=4, se_ratio=16)
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.flatten = nn.Flatten()
            self.classifier = nn.Linear(128 * 6 * 6, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.hourglass