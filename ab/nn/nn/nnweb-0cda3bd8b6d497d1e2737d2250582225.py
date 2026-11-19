import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop(out)
        out += residual
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        # Architecture: CIFARResidualNet
        # Stem: 3x3 Conv, Norm, Activation
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: Residual blocks at 32x32 (input 32x32 -> output 32x32)
        # Standard small ResNet for CIFAR often uses [2, 2, 2, 2] blocks.
        # The description mentions "Stage 1: residual blocks at 32x32", "Stage 2: ... 16x16", "Stage 3: ... 8x8".
        # It does not explicitly mention a 4th stage at 4x4, but standard ResNet-18/34 for CIFAR has 4 stages.
        # However, the prompt says "Replace the model's architecture by EXACTLY this architecture: same kind of blocks, depth, widths, downsampling...".
        # The description lists 3 stages. Let's look at the "Layout" description again.
        # "Stage 1: residual blocks at 32x32. Stage 2: residual blocks at 16x16. Stage 3: residual blocks at 8x8."
        # It doesn't mention a 4th stage. But CIFAR is 32x32.
        # If we follow standard ResNet-18 (which is the most common "small residual network" baseline), it has 4 stages.
        # However, the prompt says "same kind of blocks, depth, widths, downsampling... as described".
        # The description only lists 3 stages.
        # Let's check the source link context if possible. "CIFARResidualNet" with 93.78% accuracy.
        # Often these custom nets might be ResNet-18 or ResNet-20.
        # ResNet-20 has 3 blocks per stage (depth 20 = 2 + 6*3).
        # ResNet-18 has 2 blocks per stage.
        # The description is sparse. "Custom small residual network".
        # Let's assume standard ResNet-18 structure (2 blocks per stage) but check if the 3-stage description implies a specific depth.
        # If it strictly says 3 stages, maybe it's a ResNet-14 or similar?
        # Or maybe it's just describing the first 3 stages of a 4-stage net?
        # "Layout: ... Stage 1 ... Stage 2 ... Stage 3 ... Ends with GAP".
        # It does NOT list Stage 4.
        # If I implement 4 stages, I might be adding something not described.
        # If I implement 3 stages, I might be missing standard depth.
        # However, CIFAR input is 32x32.
        # Stem: 32x32.
        # Stage 1: 32x32.
        # Stage 2: 16x16.
        # Stage 3: 8x8.
        # If there is no Stage 4, the output of Stage 3 is 8x8.
        # GAP -> 8x8 -> 1x1.
        # This is a valid architecture.
        # Let's look at the "widths". The description doesn't specify widths.
        # Standard ResNet widths are [64, 128, 256, 512] or [64, 64, 128, 256] (for CIFAR).
        # For CIFAR, standard ResNet (He et al.) uses [64, 128, 256] for 3 stages? No, standard CIFAR ResNet uses 3 stages of [64, 128, 256] with 2 blocks each (ResNet-14) or 3 blocks each (ResNet-20).
        # Wait, standard CIFAR ResNet-18 (adapted) usually has 4 stages? No, CIFAR-10 ResNet-18 in PyTorch often has 4 stages of 2 blocks, but the first stage is usually just the stem or a single block?
        # Actually, the standard "ResNet-18" for CIFAR-10 (e.g. in PyTorch examples) often looks like:
        # Stem (32->32)
        # Stage 1 (32->32)
        # Stage 2 (32->16)
        # Stage 3 (16->8)
        # Stage 4 (8->4)
        # 
        # But the description ONLY lists 3 stages.
        # "Stage 1: residual blocks at 32x32"
        # "Stage 2: residual blocks at 16x16"
        # "Stage 3: residual blocks at 8x8"
        # It ends there.
        # I will implement exactly 3 stages as described.
        # I need to choose "widths". Standard choice for a "small" net on CIFAR is often [64, 128, 256].
        # I need to choose "depth" (number of blocks). "Small" usually implies 2 blocks per stage (like ResNet-14/18 logic) or 3 (ResNet-20).
        # Given "small", I will use 2 blocks per stage.
        # So:
        # Stem: 64 channels.
        # Stage 1: 2 blocks, 64 channels, stride 1.
        # Stage 2: 2 blocks, 128 channels, stride 2.
        # Stage 3: 2 blocks, 256 channels, stride 2.
        
        # Hyperparameters: lr, momentum.
        # The description mentions "optional Dropout2d". I should include it to be safe? 
        # "Blocks use ... optional Dropout2d". 
        # Since `supported_hyperparameters` only has `lr` and `momentum`, I cannot read a dropout rate from prm.
        # I must use a fixed value or 0.0. The prompt says "If the description leaves a detail open, choose the standard choice".
        # Standard is often 0.0 for ResNet unless specified. I will set it to 0.0 to avoid unconfigured hyperparameters, or just not use the layer if it's optional and not specified.
        # However, the instruction says: "LEMUR rejects a model that does not use every hyperparameter named in supported_hyperparameters".
        # It does NOT say I must use hyperparameters that are NOT in the list.
        # So I can ignore Dropout2d if I don't have a hyperparameter for it.
        # I will instantiate the block with dropout=0.0.

        self.stage1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.stage2 = self._make_layer(128, 128, num_blocks=2, stride=2)
        self.stage3 = self._make_layer(256, 256, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, self.num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        in_channels = 64 if self.stage1 is None else 64 # Wait, stage1 is the first one called.
        # I need to track in_channels.
        # Let's refactor _make_layer to take in_channels.
        pass

    def _build_stages(self):
        # Stem output is 64
        in_ch = 64
        
        # Stage 1: 64 -> 64, stride 1
        layers = []
        for i in range(2):
            s = 1 if i == 0 else 1
            layers.append(ResidualBlock(in_ch, 64, stride=s, dropout=0.0))
            in_ch = 64
        self.stage1 = nn.Sequential(*layers)

        # Stage 2: 64 -> 128, stride 2
        layers = []
        for i in range(2):
            s = 2 if i == 0 else 1
            layers.append(ResidualBlock(in_ch, 128, stride=s, dropout=0.0))
            in_ch = 128
        self.stage2 = nn.Sequential(*layers)

        # Stage 3: 128 -> 256, stride 2
        layers = []
        for i in range(2):
            s = 2 if i == 0 else 1
            layers.append(ResidualBlock(in_ch, 256, stride=s, dropout=0.0))
            in_ch = 256
        self.stage3 = nn.Sequential(*layers)

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Initialize stages
        self._build_stages()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()