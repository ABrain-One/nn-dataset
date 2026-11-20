from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torchvision.models._utils import _make_divisible
from torchvision.ops.misc import Conv2dNormActivation


class InvertedResidual(nn.Module):
    def __init__(
            self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


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
        
        # Architecture: 4 convolution stages, up to 256 channels, BN, ReLU, Residuals in deeper stages
        # Stage 1: 3 -> 64, 3x3, stride 1
        # Stage 2: 64 -> 128, 3x3, stride 2, Residual
        # Stage 3: 128 -> 256, 3x3, stride 2, Residual
        # Stage 4: 256 -> 256, 3x3, stride 1, Residual
        
        def conv_block(in_ch, out_ch, stride, use_residual):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            
            if use_residual:
                if in_ch != out_ch or stride != 1:
                    shortcut = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_ch)
                    )
                else:
                    shortcut = nn.Identity()
                
                return nn.Sequential(
                    nn.ModuleList([nn.Sequential(*layers), shortcut])
                )
            else:
                return nn.Sequential(*layers)
        
        # We need a custom module to handle the residual addition properly
        # Standard nn.Sequential doesn't support branching.
        class ResidualBlock(nn.Module):
            def __init__(self, in_ch, out_ch, stride):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.relu = nn.ReLU(inplace=True)
                
                if in_ch != out_ch or stride != 1:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_ch)
                    )
                else:
                    self.shortcut = nn.Identity()

            def forward(self, x):
                identity = self.shortcut(x)
                
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                
                out = self.conv2(out)
                out = self.bn2(out)
                
                out += identity
                out = self.relu(out)
                return out

        self.features = nn.Sequential(
            # Stage 1: No residual
            nn.Sequential(
                nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            # Stage 2: Residual, downsample
            ResidualBlock(64, 128, stride=2),
            # Stage 3: Residual, downsample
            ResidualBlock(128, 256, stride=2),
            # Stage 4: Residual, no downsample
            ResidualBlock(256, 256, stride=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
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

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)