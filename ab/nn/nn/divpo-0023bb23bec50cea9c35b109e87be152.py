import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self.make_layer(32, 32, 2, 1)
        self.layer2 = self.make_layer(32, 64, 2, 2)
        self.layer3 = self.make_layer(64, 128, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

        self._initialize_weights()

    def make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [Block(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(Block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
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

# Edge-friendly latency-sensitive design:
# - Use group convolutions and pointwise convolutions to keep MACs low.
# - Use block-wise expansion (e.g., 1->32->64->128->32->64->128) to maintain resolution.
# - Use depthwise separable blocks to minimize parameters and MACs.
# - Limit block depth to 3 for edge-compatibility.
# - Use efficient kernel sizes (3x3, 5x5, 7x7).
# - Ensure output feature map size is at least 6x6 for global average pooling.
# - Keep parameters below 500000 and test latency on representative edge device.

# Design suggestions:
# - Start with a stem with 32 filters and 3x3 kernels to reduce size.
# - Use 3-block (in_channels->32->64->128) paths for expansion and feature extraction.
# - Use depthwise separable blocks (in_channels->32, 32->64, 64->128, 128->32, 32->64, 64->128).
# - Use 1x1 convolutions for pointwise expansion and downsampling (32, 64, 128).
# - Use 3x3, 5x5, or 7x7 group convolutions with padding for depthwise blocks.
# - Ensure output feature maps are 6x6 for avgpool.
# - Test design on CIFAR-10 to ensure accuracy and edge latency.

# Example block (for reference):
# class Block(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))

# Net structure (for reference; adjust as needed to meet resource constraints):
# class Net(nn.Module):
#     def __init__(self, in_shape, out_shape, prm, device):
#         super().__init__()
#         self.device = device
#         in_channels = in_shape[1]  # channels-first (C, H, W)
#         num_classes = out_shape[0]
#         self.stem = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.layer1 = self.make_layer(32, 32, 2, 1)
#         self.layer2 = self.make_layer(32, 64, 2, 2)
#         self.layer3 = self.make_layer(64, 128, 2, 2)
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.classifier = nn.Linear(128 * 6 * 6, num_classes)

#         self._initialize_weights()

#     def make_layer(self, in_channels, out_channels, blocks, stride):
#         layers = [Block(in_channels, out_channels, stride=stride)]
#         for _ in range(1, blocks):
#             layers.append(Block(out_channels, out_channels))
#         return nn.Sequential(*layers)

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#