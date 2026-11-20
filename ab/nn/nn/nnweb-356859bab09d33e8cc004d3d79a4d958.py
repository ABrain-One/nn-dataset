import torch
from torch import nn, Tensor
from typing import Optional, Callable, List


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * planes)
            )

    def forward(self, x: Tensor) -> Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


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
        num_classes: int = out_shape[0]
        dropout: float = prm['dropout']

        norm_layer = nn.BatchNorm2d

        # CIFAR ResNet-18 configuration
        # Stem: 3x3 conv, stride 1, no maxpool
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)

        # Four stages of BasicBlocks
        # Stage 1: 64 channels, 2 blocks, stride 1
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1, norm_layer=norm_layer)
        # Stage 2: 128 channels, 2 blocks, stride 2
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, norm_layer=norm_layer)
        # Stage 3: 256 channels, 2 blocks, stride 2
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, norm_layer=norm_layer)
        # Stage 4: 512 channels, 2 blocks, stride 2
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, norm_layer=norm_layer)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block: Callable, planes: int, num_blocks: int, stride: int, norm_layer: Optional[Callable[..., nn.Module]] = None) -> nn.Sequential:
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        
        # Determine the input channels for the first block in this layer
        # The output channels of the previous layer are the input for this layer.
        # We can infer the previous layer's output planes based on the current layer's planes.
        # Since ResNet-18 doubles channels at each stage (64 -> 128 -> 256 -> 512):
        # Layer 1 (planes=64): input is from stem (64)
        # Layer 2 (planes=128): input is from Layer 1 (64)
        # Layer 3 (planes=256): input is from Layer 2 (128)
        # Layer 4 (planes=512): input is from Layer 3 (256)
        
        # A generic way to handle this without hardcoding the sequence is to look at the previous attribute.
        # However, since we build layers sequentially in __init__, we can just pass the previous output planes.
        # But _make_layer is called sequentially. 
        # Let's modify the logic: 
        # If planes == 64, in_planes = 64 (from stem)
        # Else, in_planes = planes // 2 (since channels double each stage in standard ResNet)
        
        if planes == 64:
            in_planes = 64
        else:
            in_planes = planes // 2

        for i, s in enumerate(strides):
            layers.append(block(in_planes, planes, s, norm_layer))
            in_planes = planes

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)