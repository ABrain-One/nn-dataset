import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32, kernel_size=3, stride=1, padding=1),
        )
        self.layer1 = nn.Sequential(
            Block(32, 32, kernel_size=3, stride=1, padding=1),
            Block(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            Block(32, 64, kernel_size=3, stride=1, padding=1, groups=32),
            Block(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            Block(64, 128, kernel_size=3, stride=1, padding=1, groups=64),
            Block(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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

# Edge-friendly latency budget (in terms of MACs): <= 40000000
# Approximated by the following multiplier (for both forward and backward passes): <= 1.0
# The actual latency is also limited by the edge device's capabilities.

# Derive the maximum number of channels and the total number of parameters
# based on the approximated multiplier (do NOT create the model yet).
# Use this information to design a model with parameters <= 500000.
# The final parameter count is the one you get after creating the model.

# The edge-friendly multiplier limits the number of MACs (Multiples of Abbreviated Convolutions) per layer.
# The MACs for a Conv2d layer are calculated as follows:
#   MACs = (C_in * H * W) * (C_out * KH * KW) + C_out * H * W
#   Here, H = 32, W = 32, KH = KW = kernel_size, C_in is the input channels.
#   The '+ C_out * H * W' term is for the bias (which is ignored in this budget).
# Use the multiplier to calculate the maximum number of output channels for each layer.
# For example, if the multiplier is 1.0, the maximum channels are:
#   out_channels = [32, 32, 64, 64, 128, 128]
# If the multiplier is 0.5, the maximum channels are:
#   out_channels = [16, 16, 32, 32, 64, 64]
# Design your model to use these channel counts, and calculate the total parameters with them.

# Example of how to use the multiplier to calculate the maximum channels:
# max_channels = [int(32 * multiplier), int(32 * multiplier), int(64 * multiplier), int(64 * multiplier), int(128 * multiplier), int(128 * multiplier)]
# max_channels = [min(32, c) for c in max_channels]  # clamp to the original channel counts
# out_channels = [min(c, max_channels[i % 6]) for i, c in enumerate([32, 32, 32, 32, 32, 32])]
# out_channels = [32, 32, 32, 32, 32, 32]  # final channel counts for each layer
# total_params = sum(c * h * w for c, h, w in [(32, 32, 32), (32, 32, 32), (32, 32, 32), (32, 32, 32), (32, 32, 32), (32, 32, 32)])

# The provided in_shape will be (3, 32, 32) for this RGB dataset.
# The final parameter count must NOT exceed 500000.
# The edge-friendly multiplier and the provided in_shape are the ONLY inputs that can be used to design the model.
# Do NOT include any other variables or constants outside the class that refer to resources (like pretrained weights)
# that are not part of the in_shape and prm parameters.

# The model MUST:
# - Use the provided in_shape and prm parameters to determine the architecture.
# - NOT use any other variables or constants (outside the class) that define resources like pretrained weights.
# - NOT use any other devices (e.g., cuda:0, cpu).
# - NOT use any other models or modules that are not a part of the Net class.
# - Use ONLY standard PyTorch layers and functions.
# - Use the provided in_shape to determine the input channels and size.
# - NOT create a model that processes inputs with a different size than [B, C, 32, 32].
# - Use the provided prm dictionary to have access to hyperparameters: lr and momentum.
# - The total parameter count MUST NOT exceed 500000.
# - MUST BE strictly under 500000 parameters.
# - MUST NOT use any part of the input or any other resource that defines more than 500000 parameters.
# - MUST be a complete and valid PyTorch model.
# - MUST NOT use any other part of the dataset or training budget beyond the provided parameters.