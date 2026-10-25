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
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = out_shape[0]

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(32, 32, 3, 1, 1)
        self.layer2 = self.make_layer(32, 64, 3, 2, 1)
        self.layer3 = self.make_layer(64, 128, 3, 2, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
        )

    def make_layer(self, in_channels, out_channels, block_num, stride, padding):
        layers = []
        layers.append(Block(in_channels, out_channels, 3, stride, 1))
        for _ in range(1, block_num):
            layers.append(Block(out_channels, out_channels, 3, 1, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# Edge-friendly latency and parameter budget.
# Choose block and layer configurations that are fast to train and low in parameters.
# Aim for an average forward time per batch (per CUDA block) below 0.5sec on V100.
# The final model parameter count must be <= 500000.
# Example block configurations (few of many possible):
#   Block(32, 32, 3, 1, 1), Block(32, 32, 3, 1, 1), Block(32, 32, 3, 1, 1)
#   Block(32, 64, 3, 1, 1), Block(64, 64, 3, 2, 1)
#   Block(32, 64, 3, 1, 1), Block(64, 64, 3, 1, 1), Block(64, 64, 3, 1, 1), Block(64, 64, 3, 1, 1)
#   Block(32, 64, 3, 1, 1), Block(64, 128, 3, 2, 1)
#   Block(32, 64, 3, 1, 1), Block(64, 128, 3, 1, 1), Block(128, 128, 3, 1, 1)
#   Block(32, 64, 3, 1, 1), Block(64, 128, 3, 1, 1), Block(128, 128, 3, 1, 1), Block(128, 128, 3, 1, 1)
#   Block(32, 64, 3, 1, 1), Block(64, 128, 3, 2, 1)
#   Block(32, 64, 3, 1, 1), Block(64, 128, 3, 1, 1), Block(128, 128, 3, 1, 1), Block(128, 128, 3, 1, 1), Block(128, 128, 3, 1, 1)

# Analyze the provided in_shape and out_shape to determine the input and output dimensions.
# Generate the first few lines of an expected valid __init__() method below.
# The target is to create a neural network with the smallest parameter count possible that meets the criteria above.
# The network should be structured as follows:
# - Input layer: Takes the input from in_shape and applies valid convolutions.
# - One or more of the following block types (define your own if needed):
#   - 2D convolution -> batch norm -> ReLU
#   - residual connections (x + conv(x)) with the same block above
#   - depthwise separable convolutions (replace or combine standard conv with depth_wise and point_wise convs)
# - An efficient classifier that maps the last layer's output to the number of classes specified by out_shape.
# The total parameter count must be <= 500000.
# No additional code outside this block is allowed.