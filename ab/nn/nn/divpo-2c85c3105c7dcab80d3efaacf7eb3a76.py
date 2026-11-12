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
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
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
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32, 32, stride=1),
            Block(32, 32, stride=1),
            Block(32, 64, stride=2),
            Block(64, 64, stride=1),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

# Edge-friendly budget: strictly under 500000 params, and latency should be under 100ms on a typical edge device.

# First, analyze the input shape and derive the correct channel count for the first conv.
# You will need to create a dummy tensor to estimate the output channels of the first block.
# Make sure to use standard layers only and do NOT hardcode any specific channel count (e.g., 32 or 64).
# The resulting model should have NO MORE THAN 500000 total parameters.

# NOTE: If the input shape is (batch, channels, height, width), then the channels-first version (C, H, W) should have channels as the FIRST dimension (C).

# Example: in_shape = [3, 32, 32] -> in_channels = 3
# Invalid: in_shape = [32, 32, 3] -> do NOT use 32 as the channel count, and do NOT hardcode it as 3

# You can use the following dummy tensors for shape calculations (and they must be defined outside the target method):
# >>> x = torch.randn(1, in_channels, 32, 32)
# >>> x = x[:, :, 3:-3, 3:-3]  # self.stem should not change input size
# >>> x.shape[1] will be the output channels after the first block.
# Use this to estimate the correct number of channels for the first conv in your design.

# The resulting model should meet the edge-friendly latency constraint.
# Benchmark on the provided dataset to ensure it runs under 100ms on a typical edge device after setup.

# The following hyperparameters can be read from prm (dictionary) if they exist in supported_hyperparameters:
# prm.get('lr', 0.01), prm.get('momentum', 0.9), and prm.get('dropout', 0.2) (if you decide to use it).

# IMPORTANT: If you need to use any additional layers or blocks from torchvision, you MUST first define them inside this scope.
# For example, to use torchvision.models.VGG_make_stage:
#   def VGG_make_stage(block, in_channels, out_channels, layers, batch_norm): ... (define entire class or function)
#   class Block_VGG(nn.Module): ... (if you adapt torchvision Block to fit your model)
# Then you can use these blocks to create your own Net class.

# The error message "VGG_make_stage is not defined" will be raised if you try to use it without an early definition.

# The resulting model should use NO MORE THAN 500000 total parameters.
# Do NOT use any existing models (including GCNs, RNNs, or any models from torchvision) with pre-existing weights, nor any models you generate with fewer than 500000 total parameters.

# The edge-friendly latency budget means that your model should run on a typical edge device within 100ms, given an input of (3, 32, 32).
# The provided dataset will be used to evaluate model accuracy.

# You MUST produce code for Net that is NOT PRESENT IN prm. NO prior runs on the data may be stored or used by your Net.

# For example, the following code is NOT allowed:
#   if 'best_acc' in prm:
#       self.some_attr = prm['best_acc']
#   elif 'best_acc' in self.__class__.__name__:
#       self.some_attr = 1.0
#   else:
#       self.some_attr = 0.0
# This would be considered a run on the data, and would be disallowed.

# The following hyperparemeters will be in prm (if present) and must be read from it if present:
# prm = {'lr': 0.01, 'momentum': 0.9, 'dropout': 0.2}
# prm = {} # edge case but could happen, your model should not require a learning rate or momentum by default

# Please print prm to verify which hyperparameters are present:
# print("prm:", prm)
# This will allow you to create edge-friendly defaults (e.g., momentum=0.9, dropout=0.2 if present).

# The resulting model must be defined entirely inside this scope, and must NOT call any methods outside this scope.