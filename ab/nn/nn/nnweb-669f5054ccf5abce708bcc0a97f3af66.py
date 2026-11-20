import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class BasicBlock(nn.Module):
    """
    Standard ResNet basic block for CIFAR (3x3 convs, no stride in first conv).
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.downsample = None
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet9(nn.Module):
    """
    ResNet-9 architecture for CIFAR-10.
    Structure:
    - Initial Conv 3x3
    - 3 Stages of 1 BasicBlock each (Total 3 blocks in residual layers + 1 initial = 4 conv layers? 
      Wait, "ResNet-9" usually implies 9 layers of weights.
      Let's look at the standard definition for CIFAR ResNet-9.
      Often it is defined as:
      Conv (1)
      Layer1: 1 Block (2 convs) -> Total 3
      Layer2: 1 Block (2 convs) -> Total 5
      Layer3: 1 Block (2 convs) -> Total 7
      Layer4: 1 Block (2 convs) -> Total 9
      
      However, standard ResNet depth naming (ResNet-110, ResNet-56) counts the total number of 3x3 conv layers.
      ResNet-9 would mean 9 conv layers total.
      Initial conv: 1
      Remaining: 8 conv layers -> 4 blocks of 2 convs each.
      So: 4 stages, each with 1 block.
      
      Let's verify widths. CIFAR ResNet typically uses widths 64, 128, 256.
      If we follow the "0.75 scale" mentioned in the prompt description, we might scale the base width.
      Standard base width is 64. 0.75 * 64 = 48.
      So widths: 48, 96, 192.
      
      Let's implement a standard ResNet with 4 stages, 1 block per stage.
      Widths: [48, 96, 192]
      
      Wait, looking at the source `randyxia07/cifar10-resnet9`:
      The repo likely defines a specific configuration.
      Common CIFAR ResNet-9 config:
      Layers: 1, 1, 1, 1 (blocks per stage)
      Widths: 64, 128, 256 (standard) or scaled.
      
      The prompt says: "configurable width (e.g., 0.75 scale)".
      I will use a base width of 64 and scale it by 0.75 to get 48.
      Widths: 48, 96, 192.
      
      Downsampling: 
      Stage 1: stride 1 (64x64 input -> 64x64)
      Stage 2: stride 2 (64x64 -> 32x32)
      Stage 3: stride 2 (32x32 -> 16x16)
      Stage 4: stride 2 (16x16 -> 8x8)
      
      This is the standard CIFAR ResNet downsampling pattern.
    """
    def __init__(self, in_channels=3, num_classes=10, width_scale=0.75):
        super(ResNet9, self).__init__()
        
        base_width = int(64 * width_scale)
        
        # Initial convolution
        self.in_planes = base_width
        self.conv1 = nn.Conv2d(in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        
        # 4 stages, 1 block each
        # Stage 1: 64x64 -> 64x64
        self.layer1 = self._make_layer(base_width, 1, stride=1)
        
        # Stage 2: 64x64 -> 32x32
        self.layer2 = self._make_layer(base_width * 2, 1, stride=2)
        
        # Stage 3: 32x32 -> 16x16
        self.layer3 = self._make_layer(base_width * 4, 1, stride=2)
        
        # Stage 4: 16x16 -> 8x8
        self.layer4 = self._make_layer(base_width * 8, 1, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, out_channels, stride))
            self.in_planes = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.channel_number = in_shape[1]
        self.image_size = in_shape[2]
        self.class_number = out_shape[0]
        
        # Use width_scale 0.75 as suggested by the description "e.g., 0.75 scale"
        self.model = ResNet9(self.channel_number, self.class_number, width_scale=0.75)

        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

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