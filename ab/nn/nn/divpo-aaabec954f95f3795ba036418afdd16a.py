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
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 64)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.avg_pool(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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


# Optional: helper blocks BEFORE Net
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


limits = 500000  # params
in_channels = 3  # from in_shape constraint

# Automatically generated size and channel choices to stay under the limit.
# You may have to revise for the best shape and channel combination.
# You can use the CIFAR10 dataset to understand the structure of the dataset and guide
# the layer selection to work with the given channel and image size constraints.
# Try to keep the image size as close to the original (32x32) as possible,
# and adapt the channel count to match the given in_channels.
channels = [32, 32, 64]  # total <= 32 (given)
kernel_sizes = [3, 3, 3]
paddings = [1, 1, 1]

# Calculated product to stay under the parameter limit.
total_params = sum((out * in_ * k * k + out) for in_, out, k in zip(channels[:-1], channels[1:], kernel_sizes))
total_params += sum(c * w * h for c, w, h in [(in_channels, 32, 32)] + [(out, 16, 16) for out in channels[1:]])
assert total_params <= limits

# Template to define the first layer with the given constraints.
# Adjust the following template to use the correct in_channels and match the image size as closely as possible.
# Try to keep the image size as close to the original (32x32) as possible.
tpl = f"self.stem = nn.Conv2d({in_channels}, 32, kernel_size=3, padding=1, bias=False)\n"