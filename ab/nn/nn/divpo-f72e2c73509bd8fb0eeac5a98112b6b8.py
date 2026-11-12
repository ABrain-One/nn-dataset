import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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

# Edge-friendly (hyper)parameters
lr = 0.025
momentum = 0.9

# Edge-friendly (hyper)parameters
lr = 0.025
momentum = 0.9

# Target edge-friendly latency (CPU inference time <= 15ms)
# Design principles:
# 1. Keep parameters below 500000 to fit on edge devices.
# 2. Prioritize channels-first convs (C x H x W) to avoid transpose ops.
# 3. Use depth (fewer layers) and width (larger channels) for efficiency.
# 4. Avoid layer combinations (e.g., Dropout followed by Conv2d).
# 5. Experiment with layer types (e.g., Attention, GhostConv).
# 6. Optimize for edge latency, not peak performance.
# 7. Use efficient activation functions (ReLU6, SiLU).
# 8. Use batch norm for early downsampling (for higher resolution).
# 9. Use depthwise convs for efficiency (like EfficientNet-lite).
# 10. Use block designs with skip connections (for lower FLOPs).
# 11. Use inverted residuals for small FLOPs per param.

# Analyze budget for edge-friendly latency (e.g., <= 15ms CPU inference).
# If FLOPs are the bottleneck, focus on channels and depth.
# If MACs are the bottleneck, use efficient activation functions.

# Analysis of the CIFAR-10 dataset:
# - Resolution: 32x32 (channels-first C x H x W)
# - Channels: 3 (RGB)
# - Size: 50,000 samples (train) + 10,000 samples (test)
# - Typical image size: ~50KB
# - Popular augmentations: crop+resize, rotate, flip
# - Edge devices are resource-constrained: RAM budget (e.g., 32MB)

# Design constraints:
# - Use standard layers only (no group norms, layer norms, etc.)
# - No pretrained weights
# - Keep parameters below 500000
# - Target edge-friendly latency <= 15ms (CPU inference)

# Design target: achieve highest accuracy with the above constraints.
# Explore layer types (e.g., attention, depthwise convs) and combinations
# that maintain edge-friendly latency.

# Optimize for channels-first convs (C x H x W) to reduce transpose ops.
# Experiment with:
# - Depth (fewer layers)
# - Width (larger channels)
# - Efficient activation functions (ReLU6, SiLU)
# - Batch normalization for early downsampling
# - Use of depthwise convs (like EfficientNet-lite)
# - Incorporation of skip connections and inverted residuals
# - Attention mechanisms (if effective with the resource constraints)

# Example design pattern (for reference, not part of the response):
# class Block(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

# Example design pattern (for reference, not part of the response):
# class DepthwiseConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = nn.ReLU(inplace=True)
#         self.expand_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False) if in_channels != out_channels else nn.Identity()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.act(x)
#         return self.expand_conv(x)

# Example design pattern (for reference, not part of the response):
# class AttentionBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // 2, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // 2, channels, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.shape
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

# The above example patterns should be adapted and combined to create a unique network design.