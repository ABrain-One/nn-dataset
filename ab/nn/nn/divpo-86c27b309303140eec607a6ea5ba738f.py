import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# This is the resulting student model.
# It should include all necessary methods (e.g., __init__, forward, train_setup, learn)
# and use them to define a complete network.
# The network should also use in_shape to derive the input channel count and not hardcode it.
# The class should be named Net.

# Example: let's create a simple model inspired by popular patterns (e.g., Resnet block).
# You can adapt this template to create your own model structure.

# Consider this as a block you might use in your model, to combine Conv2d + BatchNorm2d + Act
class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Unit, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]  # derived from in_shape, don't hardcode
        self.num_classes = out_shape[0]

        self.layer1 = Unit(self.in_channels, 32, stride=1)
        self.layer2 = Unit(32, 64, stride=2)
        self.layer3 = Unit(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            weight_decay=prm.get('weight_decay', 0.0001),
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

    def get_flops(self, input_size):
        input = torch.randn(*input_size)
        flops = torch.profiler.flops(self, input)
        return flops.to('cpu').max()