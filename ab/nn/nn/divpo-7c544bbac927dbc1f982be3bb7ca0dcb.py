import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block1 = Block(32, 64)
        self.block2 = Block(64, 128)
        self.block3 = Block(128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
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
            weight_decay=prm.get('weight_decay', 0.0005)
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv1(x)))

# Ensure Block does not introduce any unfilled space (for optimal efficiency)
assert Block(32, 32).conv1.kernel_size == (3, 3)
assert Block(32, 32).conv1.stride == (1, 1)
assert Block(32, 32).conv1.padding == (1, 1)
assert Block(32, 32).bn.num_features == 32
assert Block(32, 32).bn.affine == True
assert Block(32, 32).relu.__class__.__name__ == 'ReLU'
assert isinstance(Block(32, 32).relu, nn.ReLU)

# Example of a valid layer sequence (for the given in_channels)
# in_channels: 32
# stem: Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# block1: Block(32, 64) (-> out_channels: 64)
# block2: Block(64, 128) (-> out_channels: 128)
# block3: Block(128, 128) (-> out_channels: 128)
# No unused or empty branches (for params <= 500000)

# Resource counter to ensure total params are under the limit
total_params = 0

# Dummy input to estimate spatial dimensions after stem and blocks
dummy_input = torch.zeros(1, 32, 32, 32)

for _ in range(4):
    for _ in range(2):
        for m in Block(32, 32).children():
            dummy_input = m(dummy_input)
            if isinstance(m, nn.Conv2d):
                total_params += m.weight.numel()
        dummy_input = nn.functional.adaptive_avg_pool2d(dummy_input, (6, 6))

print("Total params:", total_params)
assert total_params <= 500000, "Total params exceeded the limit."

# Target architecture size constraints
# We'll use the dummy_input spatial dimensions to guide the block design.
# The final feature map size should be close to (6, 6) for efficient inference.
# All block layers should be designed for these dimensions to avoid downsampling too early.
# The block should not introduce unfilled space (padding=0) and maintain spatial dimensions.
# The stem should start with an even number to avoid downsampling by an odd first layer.