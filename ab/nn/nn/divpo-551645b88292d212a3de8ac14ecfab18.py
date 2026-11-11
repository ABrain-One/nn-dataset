import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 32)
        self.transition1 = Transition(32, 16)
        self.block3 = Block(16, 16)
        self.transition2 = Transition(16, 16)
        self.block4 = Block(16, 16)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(16 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.transition1(x)
        x = self.block3(x)
        x = self.transition2(x)
        x = self.block4(x)
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

# Optional: helper blocks AFTER Net, if needed by above impl
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv1(x)))

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.avgpool(x)
        return self.conv(x)


# Compute total params:
def get_n_params(model_obj):
    # lambda-wrapper to prevent closure from holding 300+MB ImageNet weights
    return sum(p.numel() for p in model_obj.parameters())

Net_base = Net(in_shape=[3, 32, 32], out_shape=[10], prm={'lr': 0.01, 'momentum': 0.9}, device='cpu')
n_accnt = 500000  # target total params
try:
    *_layers, target_layer = Net_base._modules.values()  # get last layer for budget-sensitive design
except ValueError:
    *_layers, target_layer = Net_base._modules.values()  # get last layer for budget-sensitive design

# Optimize by building towards target layer's output channels (not receptive field/padding/stride).
# Example: if target_layer should have 32x32 relu(x) at input, add <=n_accnt params to reach it.
# Resulting shape (e.g. 28x28) must be smaller than CIFAR block's max stride=2 (not 1+1, not 2+2).
# Try patterns like:
#   * AvgPool2d(2) + Conv2d->ReLU; 2x2 avgpool -> 14x14 -> 1x1 conv 32->32 -> 14x14
#   * 1x1 Conv2d 32->32 + ReLU; 1x1 32->32 convs add 32 per block to reach 32x32 in
#   * 1x1 Conv2d 32->16 + 1x1 16->32; add 32 per block to reach 32x32 in
#   * 1x1 Conv2d 32->16 + 1x1 16->32 + 1x1 32->32 + ReLU; 1x1 convs add 32 per block to reach 32x32 in
#   * 1x1 Conv2d 32->16 + 3x3 16->16 + 1x1 16->32; 1x1 convs add 16 per block to reach 32x32 in

# Analyze layers to find max receptive field for early blocks
def _rf_at_in(block):
    # Calculate receptive field assuming only 3x3 convs and same padding
    # E.g. 32x32 in -> 3x3 conv -> 30x30 out (rf=29, center 31x31 in 32x32)
    # 14x14 in -> 1x1 conv -> 14x14 out (rf=14, center 14x14 in 14x14)
    # 14x14 in -> 3x3 conv -> 12x12 out (rf=11, center 13x13 in 14x14)
    # Blocks without in_shape hint (like ones using 1x1 convs only) are allowed.
    # Return None if not sure (e.g. if blocks would not have enough params to reach target shape).
    if block._get_name() == 'Block':
        # Use provided in_channels if known; else assume 1x1 conv to reach target_layer shape
        in_channels = block.conv1.in_channels if hasattr(block, 'conv1') else 32
        return (3 + 2 * 2) * in_channels // 32
    # Add implementation for Transition and helper blocks if needed
    return None

_max_rf_at_in = max((_rf_at_in(b) for b in Net_base._modules.values() if b._get_name() == 'Block'), default=0)

# Design a single Block here to reach or exceed target_layer input size.
# Use any valid block design from above examples, plus _max_rf_at_in to limit receptive field.
# Must NOT exceed target_layer's output shape (e.g. 28x28, 16x16).
# The block must be under n_accnt; the last layer (target_layer) must be exactly at or below 500000 params.
# You can use in_shape[1] (channels=32) for the first conv's in_channels; do NOT use in_shape[0] (num_classes=32).

# Example block (you must design a new one meeting all specs above):
# class Net_Block_Template(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=