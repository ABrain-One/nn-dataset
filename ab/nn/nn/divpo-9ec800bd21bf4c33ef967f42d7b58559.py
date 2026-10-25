import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self.make_layer(32, 32, 3, 1)
        self.layer2 = self.make_layer(32, 64, 3, 2)
        self.layer3 = self.make_layer(64, 128, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(Block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(Block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
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

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Sequential()

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

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Ensure the total parameter count is <= 500000
# You can use the given in_shape and out_shape to calculate the number of output channels for the first Conv2d layer
# The output channels should be calculated based on the product of the channels and height/width after the first few layers
# The final model size should be as close to the limit as possible
# You can use the provided in_shape to derive the first conv's input channels
# The CIFAR-10 dataset uses (3, 32, 32), so you can use in_shape[1] to get the number of input channels (channels-first C x H x W)
# out_shape is (n_classes,) and should be used for the last classifier layer
# Example of how to derive the number of channels after the first few layers (you can adapt this for your architecture):
#   def derive_channels(in_shape, block_sizes, block_out_channels):
#       channels = in_shape[1]
#       for block_size, out_channels in zip(block_sizes, block_out_channels):
#           # Assuming each block downsamples by a factor of 2 (stride=2)
#           channels = (channels + 1) // 2 * out_channels
#       return channels
#   block_sizes = [3, 3, 3]
#   block_out_channels = [32, 64, 128]
#   first_conv_channels = derive_channels(in_shape, block_sizes, block_out_channels)

# Example of how to calculate the total number of parameters:
#   def count_parameters(model):
#       return sum(p.numel() for p in model.parameters() if p.requires_grad)
#   model = Net(in_shape, out_shape, prm={}, device=torch.device('cpu'))
#   total_params = count_parameters(model)