import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Use this type of block (or derive from it) for all primary blocks in the model.
# It should handle batchnorm and activation in the default implementation.
# Apply layer-norm before activation after each convolution or transformer block.
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm_layer=nn.BatchNorm2d):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(identity)
        out = self.relu(out)

        return out


# Net derivative for CIFAR-10 (channels-first C x H x W)
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]  # C
        self.image_size = in_shape[2]  # H
        num_classes = out_shape[0]  # noqa
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.dropout = prm.get('dropout', 0.3)
        self.max_epochs = prm.get('epochs', 1)
        self.max_param_count = 500000

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32, norm_layer=nn.BatchNorm2d),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32, 64, norm_layer=nn.BatchNorm2d),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=self.dropout),
            Block(64, 128, norm_layer=nn.BatchNorm2d),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=self.dropout)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.features(x)
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