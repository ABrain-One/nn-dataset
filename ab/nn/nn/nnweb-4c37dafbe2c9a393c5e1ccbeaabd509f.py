import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        
        # ResNet-9 standard configuration for CIFAR-10
        # Standard CIFAR ResNet uses BasicBlock with width 20 (base)
        # The source mentions "configurable width (e.g., 0.75 scale)".
        # Standard ResNet-9 usually refers to depth 9 (1 stem + 3 groups of 1 block? or 1 stem + 4 groups?)
        # Let's look at the specific repo structure implied by "ResNet-9".
        # Typically, ResNet depth 9 on CIFAR means:
        # Conv stem
        # 3 stages of 1 basic block each? No, that's depth 4?
        # Let's check standard definitions.
        # ResNet-18: 2 blocks per stage, 3 stages (6 blocks) + stem = depth 18? No, depth counts layers.
        # CIFAR ResNet depth formula: 6*n + 2.
        # If depth is 9, 6*n + 2 = 9 => 6n = 7, not integer.
        # However, "ResNet-9" in some contexts (like the linked repo) might refer to a specific small variant.
        # Let's assume the standard "small" ResNet for CIFAR often cited with ~94% accuracy is ResNet-14 or ResNet-20.
        # But the prompt says "ResNet-9".
        # Let's look at the description: "compact ResNet-9... configurable width... avoids aggressive downsampling".
        # If I must implement "ResNet-9", I need to interpret the depth.
        # A common interpretation for "ResNet-9" in CIFAR contexts where depth isn't strictly 6n+2 might be a variant.
        # However, looking at the provided source `randyxia07/cifar10-resnet9`, this likely refers to a specific implementation.
        # Without access to the code, I will implement a standard BasicBlock ResNet that fits the "9" label best or a standard small ResNet.
        # Actually, a very common "small" ResNet is ResNet-14 (2 blocks per stage, 2 stages? No).
        # Let's reconsider the "9" count.
        # Stem (1) + Stage 1 (1 block, 2 layers) + Stage 2 (1 block, 2 layers) + Stage 3 (1 block, 2 layers) = 1 + 2 + 2 + 2 = 7 layers?
        # If we use 2 blocks per stage:
        # Stem (1) + Stage 1 (2 blocks, 4 layers) + Stage 2 (2 blocks, 4 layers) = 9 layers?
        # This matches "ResNet-9" if we count convolutional layers in the residual blocks + stem.
        # Let's assume: Stem + 2 stages of 2 BasicBlocks each.
        # Width: Standard CIFAR ResNet uses base width 20.
        # The prompt mentions "0.75 scale" as an example of configurable width.
        # Since I must choose a standard choice where open, I will use the standard CIFAR-10 ResNet width of 20.
        # If I use width 20, the channels are [20, 40, 80].
        
        base_width = 20
        widths = [base_width, base_width * 2, base_width * 4]
        
        # Input channels
        in_channels = in_shape[1]
        
        # Stem
        self.conv1 = nn.Conv2d(in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual Blocks
        # Stage 1: 2 blocks, width base_width, stride 1
        self.layer1 = self._make_layer(base_width, 2, stride=1, in_channels=base_width)
        
        # Stage 2: 2 blocks, width base_width * 2, stride 2
        self.layer2 = self._make_layer(base_width * 2, 2, stride=2, in_channels=base_width)
        
        # Note: A "ResNet-9" might stop here or have a third stage.
        # If we count layers:
        # Conv1 (1)
        # Layer1: 2 blocks * 2 convs = 4
        # Layer2: 2 blocks * 2 convs = 4
        # Total = 9. This fits the name perfectly.
        # So we do NOT have a layer3 with stride 4? Or maybe we do?
        # If we add Layer3 (2 blocks), total becomes 13 (ResNet-14).
        # So ResNet-9 is likely Stem + 2 stages of 2 blocks.
        
        # However, standard CIFAR ResNets usually have 3 stages to downsample 32->16->8->4.
        # If we only have 2 stages, we downsample 32->32->16.
        # The description says "avoids aggressive downsampling in early layers".
        # Let's stick to the layer count interpretation: 9 conv layers.
        
        # Wait, if I use 3 stages of 1 block each:
        # Stem (1) + Stage1 (2) + Stage2 (2) + Stage3 (2) = 7.
        # If I use 2 stages of 2 blocks:
        # Stem (1) + Stage1 (4) + Stage2 (4) = 9.
        # This seems the most plausible mapping for the name "ResNet-9".
        
        # Let's verify the width. "Configurable width (e.g., 0.75 scale)".
        # Standard is 20. 0.75 * 20 = 15.
        # I will use the standard 20.
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout usage
        dropout_p = prm['dropout']
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.classifier = nn.Linear(base_width * 2, out_shape[0])

    def _make_layer(self, out_channels, num_blocks, stride, in_channels):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(in_channels, out_channels, s))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out