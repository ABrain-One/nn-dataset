import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W) — derive from in_shape
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
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

# Edge-friendly latency budget implies a smaller model and/or batch norm without scripting
# Attempt to keep params below 500000 and simplify the model where possible
# For example, no redundant skip connections, no redundant BatchNorm, etc.

# Analyze the model to ensure there are no redundant layers or batch norms
# That means the model should have:
#   NO duplicate layer names (all should be unique)
#   NO BatchNorm layers that are directly followed by another BatchNorm
#   NO BatchNorm before ReLUs that cover the entire preceding layer
# If a layer or batch norm is followed by another of the same type,
#   they should have unique names (e.g. 'block1_bn1' and 'block1_bn2')

# If the model contains BatchNorm layers that are directly followed by
# another BatchNorm or a ReLU that covers the entire preceding layer,
# simplify them to remove redundancy. BatchNorm before a ReLU should
# be removed if the ReLU covers the entire preceding layer.

# Example of redundant BatchNorm followed by another BatchNorm:
# nn.BatchNorm2d(64, momentum=0.1, eps=0.001, affine=True, track_running_stats=True),
# nn.BatchNorm2d(64, momentum=0.1, eps=0.001, affine=True, track_running_stats=True)

# Example of BatchNorm followed by a ReLU that covers the entire preceding layer:
# nn.BatchNorm2d(64, momentum=0.1, eps=0.001, affine=True, track_running_stats=True),
# nn.ReLU(inplace=True)

# Example of how to create a unique layer with a given name suffix:
# block_name = 'block1'
# block_idx = 1
# layer_name = f"{block_name}_{'bn' if use_batch_norm else 'relu'}{block_idx}"
# layer_class = nn.BatchNorm2d if use_batch_norm else nn.ReLU
# layer = layer_class(num_features, **layer_props.get(layer_name, {}))
# layer = layer.to(self.device)
# self.add_module(layer_name, layer)

# Example of how to create a unique BatchNorm2d with incrementing names:
# block_name = 'block1'
# block_idx = 1
# layer_name = f"{block_name}_{'bn1' if block_idx == 1 else 'bn2"}"
# layer_props = {
#     'block1_bn1': {'momentum': 0.1, 'eps': 0.001, 'affine': True, 'track_running_stats': True},
#     'block1_bn2': {'momentum': 0.1, 'eps': 0.001, 'affine': True, 'track_running_stats': True},
# }
# layer = nn.BatchNorm2d(64, **layer_props[layer_name])
# layer = layer.to(self.device)
# self.add_module(layer_name, layer)

# The above examples show how to create layers with unique names and
# how to handle BatchNorm2d layers that are directly followed by another
# BatchNorm2d or a ReLU that covers the entire preceding layer.
# You should modify the layer and block definitions below to adhere to
# the naming and redundancy rules specified above.

# The following maximum parameter count for the entire model is enforced:
# Current estimate: 500000
# Please ensure all layers and blocks are designed within these parameters.
#[2023/04/18 13:12:52] Case-insensitive query parameters: prm['lr'], prm['momentum']