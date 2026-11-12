import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class FastConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.init_weights = True
        self.features_extractor = self._features_extractor(prm)
        self.classifier = nn.Linear(self._get_feat_out_channels(), self.num_classes)

    def _features_extractor(self, prm):
        layers = []
        in_channels = self.in_channels
        for v in [64]:
            layers += self._conv_block(v, self._get_stride(prm), self._get_repeat(prm))
            in_channels = v
        return nn.Sequential(*layers)

    @staticmethod
    def _get_stride(prm):
        return 1

    @staticmethod
    def _get_repeat(prm):
        return 1

    def _get_feat_out_channels(self):
        return 64

    def _conv_block(self, out_channels, stride, repeat):
        strides = [stride] + [1] * (repeat - 1)
        blocks = []
        for stride in strides:
            block = FastConvBlock(self.in_channels, out_channels, stride)
            blocks.append(block)
            self.in_channels = out_channels
        return blocks

    def forward(self, x):
        x = self.features_extractor(x)
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        x = avgpool(x)
        x = self.classifier(x.squeeze())
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.)
            self.optimizer.step()