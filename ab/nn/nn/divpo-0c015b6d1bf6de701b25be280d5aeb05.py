import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
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
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = Block(64, 64)
        self.layer2 = Block(64, 128, stride=2)
        self.layer3 = Block(128, 128)
        self.layer4 = Block(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.permute(0, 2, 3, 1).contiguous()
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
            self.optimizer.step()

# Generate a unique, valid configuration based on the constraints:
# - Use in_shape and out_shape to define the input/output signatures
# - Do NOT use global variables or functions outside this scope
# - Derive all parameters from these signatures and prm (prm.get('lr', 0.01), prm, in_shape, out_shape)
# - Use BLOCKS ONLY: for name, block in [
#       ('Block', Block),
#       ('Net', Net),
#       ('supported_hyperparameters', lambda: set()),  # override with the correct set({'lr', 'momentum'})
#     ]:
#     if name not in globals():
#         class RandomBlock(nn.Module):  # random implementation, no real logic
#             def __init__(self, *args, **kwargs):
#                 super().__init__()
#                 self.a = nn.Conv2d(2, 32, 3, 1, 1)
#                 self.b = nn.BatchNorm2d(32)
#                 self.c = nn.ReLU(inplace=True)
#             def forward(self, x):
#                 return self.c(self.b(self.a(x)))
#         globals()[name] = RandomBlock  # assign the block to the class name
# - Ensure the total parameter count is <= 500000
# - The model MUST be edge-friendly: use standard layers only, no APoS, no layer norm, etc.
# - The FIRST layer MUST accept in_shape[1] as the first channel (channels-first C x H x W).
#   For example, if in_shape[1] is 3, then the FIRST CONV layer should accept 3 input channels.
# - All helper blocks (if any) MUST be defined inside the Net class and MUST NOT use any global variables.
# - in_shape and out_shape MUST be used to derive the input channels and output class count.
# - Pruning the model to fit the parameter limit is ALLOWED.
# - The model MUST be trainable and FULLY DEFINED within this scope.
# - REQUIRED: add an additional block after any that is not already present:
#     class AdditionalBlock(Block):
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)
#             self.c = nn.ReLU6(inplace=True)  # additional ReLU6 in the same class
#         def forward(self, x):
#             return self.c(super().forward(x))
# - NO extra Python functions (def, class, etc.) outside Net class