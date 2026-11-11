import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.conv(x) + self.shortcut(x))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=1)
        self.layer1 = Block(32, 32, stride=1)
        self.layer2 = Block(32, 64, stride=2)
        self.layer3 = Block(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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
            self.optimizer.step()

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}

# Edge-friendly layer: AdaptiveAvgPool2d(6, 6) -> reduce to 1024 channels
#                        (6*6*64 -> 4096 -> 1024 -> 10 classes)
# Keep params below 500k: prefer depth, but may need to tweak channel counts:
#   - 32 -> 32 -> 64 -> 128 -> 4096 -> 1024 (keep <= 32, sum to <= 500k)
# Avoid common pitfalls:
#   - No residual/skip connections (no F.relu(x + self.conv(x)) or x + self.shortcut(x))
#   - No inplace ops: ReLU(inplace=True) --> ReLU()
#   - No batchnorm before activation (BatchNorm2d after Conv2d)
#   - No group norm, no depthwise conv, no advanced layer outs
#   - No SE blocks, no Ghost modules, no ResNeSt, etc.
#   - Keep layer count to <= 11 (10 => first/last are for stem and classifier)
# Focus on latency: edge-friendly shapes (64->32, 64, 64->32, 128)
#                   --> prefer 32x32 stem, keep 32->32->64->128 chains short
#                   --> no Dilated/StridedConv, NoNL, SPP, Large kernels
# Resource close to edge: batchnorm memory, no in1t, efficient configs