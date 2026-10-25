import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv1(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]   # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self.make_layer(32, 32, 2, stride=1)
        self.layer2 = self.make_layer(32, 64, 2, stride=2)
        self.layer3 = self.make_layer(64, 128, 2, stride=2)
        self.layer4 = self.make_layer(128, 256, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
        )

    def make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [Block(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(Block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
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

# Edge-friendly latency optimization suggestions (you can ignore if not applicable):
# 1. Avoid layers/operations not supported by edge devices.
# 2. Use groups in convolutions to keep memory usage low.
# 3. Balance layer dimensions to keep MACs/params low.
# 4. Use depthwise separable convolutions (groups=in_channels).
# 5. Use efficient batch norm implementations (e.g., SyncBatchNorm).
# 6. Prefer pre-activation blocks (Residual + Conv).
# 7. Use 3x3 convolutions instead of 1x1s for cheap expansion.
# 8. Use pointwise convolutions for cheap expansion.
# 9. Use separable convolutions for cheap expansion.
# 10. Use depthwise convolutions for cheap expansion.
# 11. Use kernel sizes 3, 5, or 7 for cheap expansion.
# 12. Balance out expansion ratios for cheap expansion.
# 13. Use pointwise convolutions for cheap reduction.
# 14. Use separable convolutions for cheap reduction.
# 15. Use depthwise convolutions for cheap reduction.
# 16. Use 3x3 convolutions instead of 1x1s for cheap reduction.
# 17. Balance out reduction ratios for cheap reduction.
# 18. Use stride=2 for cheap downsampling (if needed).
# 19. Use efficient upsampling/interpolation methods.
# 20. Use low-precision data types (e.g., qint8).
# 21. Use fused batch norm and activation ops when available.
# 22. Use low-rank factorization for convolutions.
# 23. Use separable transforms for convolutions.
# 24. Use group convolutions for parallelism.
# 25. Use pointwise convolutions for cheap pointwise ops.
# 26. Use efficient pointwise implementations (e.g., Conv1x1).
# 27. Use efficient depthwise implementations (e.g., DepthwiseConv3x3).
# 28. Use efficient pointwise-depthwise combinations.
# 29. Use efficient separable convolutions (e.g., DepthwiseSeparableConv3x3).
# 30. Use efficient reduction paths (e.g., 1x1->3x3->1x1).