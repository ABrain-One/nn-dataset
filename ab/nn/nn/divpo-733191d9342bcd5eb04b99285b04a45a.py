import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer1 = self._make_layer(32, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, out_shape[0])

    def _make_layer(self, inplanes, planes, stride):
        layers = []
        layers.append(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

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
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0001)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()


params = 500000
module_name = Net.__name__

# Design criteria:
# 1. Achieve high accuracy with the given dataset.
# 2. Tune for latency: use efficient block designs.
# 3. Use edge-friendly activation functions (RELU6).
# 4. Keep parameters under the limit (params <= 500000).

# Example block (for reference, do NOT use in design):
class MobileInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(MobileInvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # Pointwise
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),

            # Depthwise
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),

            # Pointwise
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

                   
# Design skeleton (fill in the empty block below with your block):
class Block:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# The following block is used to generate the architecture.
class Candidate:
    def __init__(self, block):
        self.block = block

    def __call__(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            self.block(in_channels, out_channels, stride=stride),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def get_out_channels(self):
        return self.block.block[0].out_channels

block = Block(in_channels=32, out_channels=32)
candidate = Candidate(block)

# Target parameters and latency budget:
# Parameters should not exceed the given limit: params <= 500000.
# Design should optimize for latency, adhering to edge-friendly activation functions (RELU6).
# Keep the maximum channel count (after block) below 64.
# Choose block and layer dimensions based on early max-pooling or spatial size reduction.