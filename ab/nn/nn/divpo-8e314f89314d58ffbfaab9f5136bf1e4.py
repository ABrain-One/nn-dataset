import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        self._is_fwd_compatible_with_ddp = True

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
        )
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
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

# Edge-friendly latency is the most important constraint.
# Use conv channels and kernel_size to achieve the highest possible FV/params.
# For example, consider using 1x1 convolutions or 3x3 convolutions with dilation=1,2,3.
# The target FV is 79.2 (channel-wise mean). Target params <= 500000.
# Think about global pooling, attention mechanisms, or high-frequency patterns.

# Analysis: The dataset has 32x32 images in RGB format.
# The target is to achieve a high feature variance (FV) while staying under the param limit.
# Consider using 1x1 or 3x3 convs with strides=1,2,3 for high-freq patterns.
# Incorporate attention mechanisms for higher FV. Global pooling (e.g., GAP) helps extract spatial patterns.

# Example attention mechanisms: Channel Attention, Spatial Attention, or Multihead Self-Attention.
# Consider using 1x1 convolutions for channel-wise attention.
# Experiment with dilated convolutions (dilation=1,2,3) for higher frequency patterns.

# Due to the edge-friendly latency constraint, optimize for minimal computation and maximum channel parallelism.

# Generate a unique architecture with the following guidelines:
# - Use channel parallelism and high-frequency patterns to achieve high FV.
# - Prefer 1x1 or 3x3 convolutions with strides=1,2,3.
# - Incorporate global pooling or attention mechanisms for effective feature extraction.
# - The total parameter count must be <= 500000.
# - Design for minimal computation to meet the edge-friendly latency constraint.

# Think critically about the interdependencies between channels and spatial dimensions.
# Experiment with dilated convolutions (dilation=1,2,3) for high-frequency patterns and channel-wise attention.
# The generated architecture must be original and not copied from the training set.

# Evaluate the design's performance in terms of FV and computation complexity.
# No models or blocks may be directly copied from the training set or external libraries.