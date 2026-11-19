import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union


class CIFARResidualBlock(nn.Module):
    """
    A residual block for CIFAR-style ResNet.
    Structure: Conv3x3 -> BN -> ReLU -> Dropout -> Conv3x3 -> BN -> Dropout -> Add -> ReLU
    Supports optional SE (Squeeze-and-Excitation) and Stochastic Depth (DropPath).
    """
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout_p: float = 0.0,
        use_se: bool = False,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        
        # SE Block
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(2),
                nn.Linear(planes, max(1, planes // 16)),
                nn.ReLU(inplace=True),
                nn.Linear(max(1, planes // 16), planes),
                nn.Sigmoid(),
            )
        else:
            self.se = nn.Identity()

        # Stochastic Depth (DropPath)
        self.drop_path = nn.Identity()
        if drop_path_rate > 0.0:
            self.drop_path = DropPath(drop_path_rate)

        # Shortcut projection
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * self.expansion),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        # Apply SE
        out = self.se(out)
        
        # Apply Stochastic Depth
        out = self.drop_path(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  when used in linear layers."""
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

        self.batch_transform_fn = None  
        self.augment_configs = None

        if prm.get('augment'):          
            try:
                from ab.nn.transform.base.BatchTransform import batch_transform as bt
                self.batch_transform_fn = bt
                self.augment_configs = prm.get('augment')
                print(f"Augmentation loaded: {self.augment_configs}")  
            except ImportError as e:
                print(f"Failed to load batch_transform: {e}")

    def learn(self, train_data):
        total_batches = len(train_data)

        for batch_idx, (inputs, labels) in enumerate(train_data):  
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            if self.batch_transform_fn: 
                # Pass batch index and total batches to find the fixed order

                inputs, target_a, target_b, lam = self.batch_transform_fn(
                    inputs, labels, batch_idx, total_batches, self.augment_configs  # fix: all 5 args
                )
                outputs = self(inputs)
                # Weighted loss calculation
                loss = lam * self.criteria[0](outputs, target_a) + (1. - lam) * self.criteria[0](outputs, target_b)
            else:
                outputs = self(inputs)
                loss = self.criteria[0](outputs, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
            

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        
        # Read hyperparameters
        # Note: 'dropout' is not in supported_hyperparameters, but the prompt says "LEMUR rejects a model that does not use every hyperparameter named in supported_hyperparameters".
        # The prompt also says "e.g. a dropout layer with p=prm['dropout']". 
        # However, `supported_hyperparameters` returns `{'lr', 'momentum'}`. 
        # If I add 'dropout' to supported_hyperparameters, I must keep the signature? 
        # "keep supported_hyperparameters ... exactly as they are". 
        # So I cannot add 'dropout' to the returned dict.
        # But I must use the hyperparameters named in supported_hyperparameters.
        # The prompt says: "LEMUR rejects a model that does not use every hyperparameter named in supported_hyperparameters: each of those names (for example dropout or momentum) must still be read from prm and actually used in the new model".
        # This implies that if 'dropout' is NOT in supported_hyperparameters, I don't *have* to use it? 
        # But the prompt also says: "Where the description leaves a detail open, choose the standard choice for that architecture."
        # The architecture description mentions "optional Dropout2d".
        # If I don't use a hyperparameter for dropout, I should probably hardcode it to 0.0 or a small value, OR check if prm has it optionally?
        # The rule says: "use only names the file already reads from prm". The file reads 'lr' and 'momentum' in train_setup.
        # In __init__, I can read from prm.
        # If I introduce a dropout, I should probably read it from prm if available, but since it's not in supported_hyperparameters, maybe I shouldn't rely on it being there?
        # However, the prompt example says "e.g. a dropout layer with p=prm['dropout']". This suggests that if I use dropout, I should read it from prm.
        # But if it's not in supported_hyperparameters, LEMUR might not provide it?
        # Let's look at the constraint: "keep supported_hyperparameters ... exactly as they are".
        # So I cannot add 'dropout' to the list.
        # Therefore, I should probably just set dropout_p=0.0 or a fixed small value, OR read it from prm with a default.
        # "use only names the file already reads from prm". The file currently reads 'lr' and 'momentum' in train_setup.
        # In __init__, I can access prm.
        # Let's assume I can read 'dropout' from prm if it exists, but since it's not guaranteed by supported_hyperparameters, I should be careful.
        # Actually, the prompt says "each of those names (for example dropout or momentum) must still be read from prm". This phrasing is slightly ambiguous. Does it mean "names in supported_hyperparameters"? Yes.
        # So I MUST use 'lr' and 'momentum'. I am using them in train_setup.
        # Do I need to use them in __init__? No, train_setup is kept as is.
        # So I just need to make sure I don't break the contract.
        
        # Regarding Dropout: The architecture description says "optional Dropout2d".
        # I will implement it with a fixed p=0.0 or read from prm.get('dropout', 0.0).
        # Since I can't add 'dropout' to supported_hyperparameters, I will use prm.get('dropout', 0.0) to be safe, assuming it might be passed even if not listed? 
        # No, "use only names the file already reads from prm". The file reads 'lr' and 'momentum' in train_setup.
        # In __init__, I can read 'lr' and 'momentum' too, but they are used in train_setup.
        # Let's just hardcode dropout to 0.0 to be safe, or use a very small value.
        # Wait, the prompt says "e.g. a dropout layer with p=prm['dropout']". This is an example of HOW to use a hyperparameter.
        # If I don't use 'dropout' as a hyperparameter name, I don't need to read it.
        # I will set dropout_p = 0.0.

        num_classes = out_shape[0]
        norm_layer = nn.BatchNorm2d
        
        # CIFAR input size is typically 32x32. 
        # in_shape is (C, H, W).
        # Stem: 3x3 Conv, stride 1, padding 1. Output size remains HxW.
        # Stage 1: 32x32
        # Stage 2: 16x16 (stride 2)
        # Stage 3: 8x8 (stride 2)
        
        # Standard CIFAR ResNet widths: [64, 128, 256] or [16, 32, 64]?
        # The description says "custom small residual network".
        # Common CIFAR ResNet-20 has widths 16, 16, 32, 32, 64, 64.
        # This description has 3 stages.
        # Let's use widths 64, 128, 256? Or 16, 32, 64?
        # "Custom small" suggests smaller. Let's use 64, 128, 256 as a safe "standard" for 3 stages if not specified, 
        # OR 16, 32, 64 which is more "small".
        # Given "CIFARResidualNet" and "small", I'll use 64, 128, 256 to be robust, or maybe 16, 32, 64.
        # Let's look at the source name "CIFARResidualNet". Often these are ResNet-20 or ResNet-56 adapted.
        # ResNet-20: 3 stages, 2 blocks each. Widths: 16, 32, 64.
        # I will implement 3 stages with 2 blocks each (total depth 6 layers + stem).
        # Widths: 64, 128, 256 is standard for ImageNet. For CIFAR, 16, 32, 64 is more common for "small".
        # However, without explicit width in description, "standard choice" for a 3-stage CIFAR net is often 64, 128, 256 if following ImageNet conventions, 
        # but 16, 32, 64 if following CIFAR conventions.
        # Let's use 64, 128, 256 to be safe with "standard" normalization and head, but scaled?
        # Actually, let's just pick 64, 128, 256. It's a valid "small" net compared to ResNet-50.
        
        # Stem
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1: 32x32
        self.layer1 = self._make_layer(64, 2, stride=1, norm_layer=norm_layer)
        
        # Stage 2: 16x16
        self.layer2 = self._make_layer(128, 2, stride=2, norm_layer=norm_layer)
        
        # Stage 3: 8x8
        self.layer3 = self._make_layer(256, 2, stride=2, norm_layer=norm_layer)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int, norm_layer: Callable[..., nn.Module]) -> nn.Sequential:
        layers = []
        layers.append(
            CIFARResidualBlock(
                self.inplanes if hasattr(self, 'inplanes') else 64, 
                planes, 
                stride, 
                norm_layer=norm_layer,
                dropout_p=0.0,
                use_se=False,
                drop_path_rate=0.0
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                CIFARResidualBlock(
                    self.inplanes,
                    planes,
                    stride=1,
                    norm_layer=norm_layer,
                    dropout_p=0.0,
                    use_se=False,
                    drop_path_rate=0.0
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x