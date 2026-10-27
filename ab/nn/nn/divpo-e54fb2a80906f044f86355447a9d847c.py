import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = Block(in_channels=in_shape[1], out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer1 = Block(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer2 = Block(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.layer3 = Block(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.layer4 = Block(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, out_shape[0])
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = None

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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

# Ensure the maximum number of parameters is not exceeded
def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def prm_to_device(prm):
    return prm.get('device', 'cpu')

# Ensure the maximum number of parameters is not exceeded
def get_maximum_parameters(in_shape, out_shape, prm):
    device = prm_to_device(prm)
    model = Net(in_shape, out_shape, prm, device)
    return get_parameter_count(model)

# Dummy in_shape to pass the shape validation check (actual value does not matter)
in_shape = (3, 32, 32)
out_shape = (10,)
prm = {'device': 'cpu', 'lr': 0.01, 'momentum': 0.9}
max_params = 500000

# Find the maximum number of parameters within the constraint
def find_maximum_parameters(in_shape, out_shape, prm):
    device = prm_to_device(prm)
    max_params = 500000
    best_params = 0
    best_layer = None

    for _ in range(10):  # Run a few iterations to find a good block configuration
        layer = Block(
            in_channels=32,  # Randomly choose input channels
            out_channels=32,  # Randomly choose output channels
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Compute the number of parameters for the current layer
        layer_params = get_parameter_count(layer)

        # If the layer fits within the parameter limit, try to fit it into the model
        if layer_params + best_params <= max_params:
            net = Net(in_shape, out_shape, prm, device)
            net_params = get_parameter_count(net)

            # If the entire model fits within the parameter limit, update the best parameters
            if net_params <= max_params:
                best_params = net_params
                best_layer = layer

    # Return the best block configuration found
    return best_layer

# Define the block with the best parameters found
best_block = find_maximum_parameters(in_shape, out_shape, prm)

# Define the final Net class with the best block and the rest of the architecture
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = Block(
            in_channels=in_shape[1],
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.layer1 = Block(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.layer2 = Block(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.layer3 = Block(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.layer4 = Block(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = None

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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

supported_hyperparameters = {'lr', 'momentum'}