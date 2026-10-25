import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self._make_layer(32, 32, 3, 1, 1)
        self.layer2 = self._make_layer(32, 64, 3, 2, 1)
        self.layer3 = self._make_layer(64, 128, 3, 2, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def _make_layer(self, in_planes, out_planes, blocks, stride, padding):
        layers = [Block(in_planes, out_planes, stride=stride, padding=padding)]
        for _ in range(1, blocks):
            layers.append(Block(out_planes, out_planes, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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
            self.optimizer.step()

# Dummy input to derive the correct in_channels and out_shape
dummy_input = torch.randn(1, 3, 32, 32)
in_channels = dummy_input.shape[1]
out_shape = (10,)

# Standard value for the number of parameters
estimated_params = 500000

# Design criteria
maximize_accuracy = True
edge_friendly = True

# Helper function to calculate the number of parameters
def calculate_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Target number of parameters
target_params = estimated_params

# Edge-friendly and accuracy-oriented early-stopping threshold
edge_threshold = 0.01 if edge_friendly else 0.05
accuracy_threshold = 0.9 if maximize_accuracy else 0.85

# Attempt to fit within the parameter limit with early stopping
early_stop = False

# Run the design process
try:
    # Dummy forward pass to set the module names for the pruner
    _ = Net(dummy_input.shape, out_shape, {'lr': 0.01, 'momentum': 0.9}, device='cpu').eval()

    # Actual design process
    pruner = torch.quantization.utils.get_default_qconfig('fbgemm').prune_function
    for module_name in ['layer1', 'layer2', 'layer3']:
        module = Net.named_children().__next__()[1].get_submodule(module_name)
        module = pruner(module, {'p': 0.5}, False)  # prune 50% of the channels
        estimated_params = calculate_params(module)
        if estimated_params >= target_params:
            break
        if estimated_params < target_params * (1 - edge_threshold):
            early_stop = True
        if not early_stop and not maximize_accuracy:
            module.apply_pruning()  # apply pruning and reduce the number of channels
        if early_stop or (maximize_accuracy and module_name == 'layer3'):
            break
except Exception as e:
    print(f"Design failed with error: {e}")

# The actual model after pruning or the best attempt before exceeding parameters
model = Net(dummy_input.shape, out_shape, {'lr': 0.01, 'momentum': 0.9}, device='cpu')
model.train_setup({'lr': 0.01, 'momentum': 0.9})