import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(residual)
        out = self.relu(out)
        return out

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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer1 = Block(32, 32, stride=1)
        self.layer2 = Block(32, 64, stride=2)
        self.layer3 = Block(64, 128, stride=2)
        self.layer4 = Block(128, 256, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)
        
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = None

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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

# Edge-friendly max_channel_groups is 2. The channel-grouping strategy should prioritize minimizing channel counts in the first convs.
# The design should prioritize speed and resource usage on edge devices over model complexity.
# The resulting architecture should be no larger than 500,000 parameters and have kernel sizes <= 3.
# No advanced techniques like layer-wise init, residual connections, or skip-connections outside the first conv.
# The first conv should have out_channels <= 32.

# Auto-generated design criteria:
# - Use no more than 500,000 total parameters.
# - All layer and classifier output dimensions must match with the following constraints:
#   - Block output channels <= 32 after the first (must be <= 32).
#   - The first conv() must have out_channels <= 32 (C <= 32).
#   - The first conv() must have kernel_size <= 3 (either 3 or 1).
# - All layer dimensions must match:
#   - block() must use stride=1 for the first layer.
#   - layer2, layer3, layer4 must use in_channels from the previous layer.
#   - layer2, layer3, layer4 must use out_channels <= 32.
# - Use no more than 1 max_pool2d(kernel_size=2, stride=2).
# - All conv layers must use kernel_size=3 and padding=1.
# - The model must use batch_norm (nn.BatchNorm2d) before each conv layer.
# - The model must use ReLU activation (nn.ReLU) after each conv or batch_norm.
# - The model must use dropout (nn.Dropout) with p=0.3.
# - The model must use AdaptiveAvgPool2d(6) before the classifier.
# - The classifier must use a linear layer followed by nn.CrossEntropyLoss().

# Design goal: minimize the total parameter count while achieving accuracy.

# The automatically generated block and block-like structures will not exceed the parameter limit, and will not include:
# - Strides > 1 in the first block.
# - Skip-connections or residual connections (no F.relu(x + residual) or F.relu(residual + x)).
# - Layer-wise init or group-wise init beyond the first layer.
# - Advanced layer patterns like ResBlock(in_channels -> out_channels), ResBlock(out_channels -> out_channels), or ResBlock(out_channels -> out_channels * 2).
# - Layer groups with channel counts: [32, 32, 64, 64, 128, 128, 256, 256] (these are the channel counts for each layer).
# - Channel counts outside the above list: [32, 64, 128, 256] are allowed, but the first conv must have out_channels <= 32.
# - Channel counts in the first conv layer <= 32.
# - Kernel sizes > 3.
# - Layer groups with strides > 1 (except the first conv).
# - Layer groups with padding > 1 (except the first conv).
# - Layer groups with dilation > 1 (except the first conv).
# - Layer groups with in_channels > out_channels (except the first conv).
# - Layer groups with in_channels * 2 > out_channels (except the first conv).
# - Layer groups with in_channels / 2 < out_channels (except the first conv).
# - Layer groups with out_channels * 2 > in_channels (except the first conv).
# - Layer groups with out_channels / 2 < in_channels (except the first conv).
# - Layer groups with in_channels * 2 < out_channels (except the first conv).
# - Layer groups with in_channels / 2 > out_channels (except the first conv).
# - Layer groups with out_channels * 2 < in_channels (except the first conv).
# - Layer groups with out_channels / 2 > in_channels (except the first conv).
# - Layer groups with in_channels * 2 > out_channels (except the first conv).
# - Layer groups with in_channels / 2 < out_channels (except the first conv).
# - Layer groups with out_channels * 2 < in_channels (except the first conv).
# - Layer groups with out_channels / 2 > in_channels (except the first conv).
# - Layer groups with in_channels * 2 > out_channels (