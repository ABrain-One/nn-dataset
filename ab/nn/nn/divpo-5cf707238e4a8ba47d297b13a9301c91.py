import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # derive from in_shape, DO NOT use in_shape[0]
        num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.2))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 1e-4)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()


bounds = 500000  # total parameters

# Generate a model that fits bounds by:
# 1. Start with a simple block (Blocks 1-3 below).
# 2. Make the number of output channels a multiple of 16 (e.g., 32, 64, 160).
# 3. Use a channel multiplier that lands the model at or under bounds
#    (e.g., 1.0, 1.25, 1.5, 1.75).
# 4. Use blocks with kernel sizes (3, 3, 5).
# 5. Make the block stem width a multiple of 16 (e.g., 16, 32, 64).

# Example block (for reference).
class Block(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, stride, padding):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Target: total_params <= 500000.
# Goal: minimize final_layer_size * num_classes.
# Hyperparameters: learning_rate=0.01, momentum=0.9.

# Analyze bounds and derive the size of the first layer:
class Analyze(nn.Module):
    def __init__(self, ch=32, c=3):
        super(Analyze, self).__init__()
        self.target = Net(in_shape=(c, 32, 32), out_shape=(1,), prm={'lr': 0.01, 'momentum': 0.9}, device='cpu')
        self.stem = Block(c, ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Run through the first block to determine the output size
        x = self.stem(x)
        # Check the output size and calculate the product with num_classes
        size = x.data.shape[1] * x.data.shape[2] * x.data.shape[3]
        return size
    def get_memory(self):
        x = torch.zeros(2, *self.stem.conv.weight.shape[1:], device='cuda')
        return self.forward(x).element_size() * self.target.classifier.weight.element_size()
    def get_params(self):
        return sum(p.numel() for p in self.target.parameters())

class Solution(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Solution, self).__init__()
        self.analyze = Analyze(ch=32, c=in_shape[1])
        self.net = Net(in_shape=in_shape, out_shape=out_shape, prm=prm, device=device)
        self.net.classifier.in_features = self.analyze.get_memory()
        self.net.classifier.out_features = out_shape[0]
        self.total_params = self.analyze.get_params()