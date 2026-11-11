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
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W) — DO NOT hardcode
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, stride=1, padding=1),
            Block(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = Block(32, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = Block(32, 64, kernel_size=3, stride=1, padding=1)
        self.layer3 = Block(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer4 = Block(64, 128, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
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

# Edge-friendly, channels-first (C, H, W), max shape [64, 32, 32]
# Ensure maximum width and height of the feature maps is 32 after all layers
# Maximal parameter count is 500000
# Target throughput is 500-700 samples/sec on Jetson Nano

# Auto-generated analysis of the dataset's shape, may be inaccurate:
#   channels_first: True
#   channels: 3
#   height: 32
#   width: 32
#   dtype: float32
#   device: cpu
#   augmentations:
#     RandomResizedCrop(size=32, scale=(0.75, 1.0))
#     RandomHorizontalFlip()
#     ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)
#     RandomGrayscale(p=0.1)
#     ToTensor()
#     Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

# Analyzed target train batch size: 128 (actual may be smaller due to augmentation and/or clipping)
# Target updates per second: 1.50 (average, consider introducing LR decay for more)
# Target iterations on dataset: 3.00 (epochs)
# Target total training time on Jetson Nano: 30 minutes
# Target latency per operation: 2000 us (minimum), 3000 us (maximum)
# Target parameters: <= 500000
# Target FLOPs (edge-friendly): 10000000000000.00

# Analyze the dataset's statistics and select an architecture that:
# - Is edge-friendly and respects the latency budget.
# - Maintains channels-first (C, H, W) ordering.
# - Does not use pretrained weights.
# - Adheres to the parameter count limit of 500000.
# - Has a max shape of [64, 32, 32] after all layers.
# - Does not use AdaptiveAvgPool2d(6) or larger due to latency.
# - Does not use SE blocks.
# - Uses 32x32 convolutions only.
# - Does not use GELU or Hardswish activations.
# - Prefers depth (number of layers) over width (channel count).
# - Does not use layers outside the given block template.
# - Does not use in_channels or out_channels < 32.
# - Does not use zero-padding > 1.
# - Does not use stride > 2.
# - Does not use dilation > 1.
# - Does not use output kernels larger than 3x3.
# - Does not use padding-less convolutions.
# - Prefers 1x1 -> 3x3 -> 1x1 convolutions over 3x3 -> 3x3.
# - Does not use ResNext-style blocks with r >= 2.
# - Uses standard layers only:
#   B0: Block (self)
#   B1: Block (self), skip B0
#   B2: Block (self), skip B1
#   B3: Block (self), skip B2
#   B4: Block (self), skip B3
# - Does not use any blocks or modules outside the given template (e.g. No BN-ACT, NoSeparableConv, NoAttention).
# - Does not use any auxiliary heads, loss functions, or loss-specific layers (such as BCEWithLogitsLoss, FocalLoss, TripletMarginLoss).
# - Does not use Dropout or DropPath with p >= 0.5.
# - Does not use MultiLabelSoftMarginLoss (for single-label tasks).
# - Does not use LayerNorm or GroupNorm (consider using BatchNorm2d).
# - Does not use GroupedConv2d or Conv2d groups.
# - Does not use AvgPool2d with kernel > 3 or stride > 1.
# - Prefers batch norm over GroupNorm or LayerNorm.
# - Does not use inplace operations (relu6(inplace=True) is allowed).
# - Uses LeakyReLU with negative_slope=0.1 or inplace=True.
# - Does not use inplace operations after any activations (LeakyReLU, SELU, etc.).
# - Does not use inplace transpose convolutions (F.conv_transpose2d with inplace=True).
# - Does not use inplace activations (relu(inplace=True) or hardtanh(inplace=True)).
# - Does not use inplace batch norm (bn(inplace=True)).
# - Does not use inplace operations outside supported ops: relu, hardtanh, leaky_relu, elu, selu, prelu.
# - Prefers depth-wise separable convolutions over regular convolutions.
# - Prefers convolutions with kernel size 3 or 5 over 1.
# - Does not use in_2x2_groups_out or in_3x3_64_groups_out.
# - Prefers 3x3 convolutions over 1x1 convolutions (unless stride > 1).
# - Prefers 3x3 convolutions with padding 1 over 5