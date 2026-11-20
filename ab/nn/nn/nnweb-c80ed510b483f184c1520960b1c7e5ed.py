from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

InceptionOutputs = namedtuple("InceptionOutputs", ["logits", "aux_logits"])
InceptionOutputs.__annotations__ = {"logits": Tensor, "aux_logits": Optional[Tensor]}

_InceptionOutputs = InceptionOutputs


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
        transform_input: bool = False
        inception_blocks: Optional[List[Callable[..., nn.Module]]] = None
        init_weights: Optional[bool] = None
        dropout: float = prm['dropout']
        if inception_blocks is None:
            inception_blocks = [BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux]
        if init_weights is None:
            init_weights = True
        if len(inception_blocks) != 7:
            raise ValueError(f"length of inception_blocks should be 7 instead of {len(inception_blocks)}")
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = False
        self.transform_input = transform_input
        # Reduce initial channel width to speed up convergence and reduce overfitting risk in 1 epoch
        self.Conv2d_1a_3x3 = conv_block(in_shape[1], 16, kernel_size=3, stride=1)
        self.Conv2d_2a_3x3 = conv_block(16, 16, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = conv_block(16, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(32, 40, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(40, 96, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Custom lightweight Inception blocks for CIFAR-10 (32x32 input)
        # After maxpool2, spatial size is 8x8. We need to handle this carefully.
        # Standard Inception expects larger inputs. We will use smaller channel counts.
        
        # Block 1: 8x8 input
        self.Mixed_5b = inception_a(96, pool_features=16)
        # Output channels: 32+32+48+16 = 128.
        
        self.Mixed_5c = inception_a(128, pool_features=32)
        # Out: 32+32+48+32 = 144.
        
        self.Mixed_5d = inception_a(144, pool_features=32)
        # Out: 32+32+48+32 = 144.
        
        # Transition 6a: 256 input.
        # InceptionB outputs: 384 (3x3 stride 2) + 96 (3x3dbl stride 2) + 256 (pool stride 2) = 736.
        # This is too wide for 8x8 spatial? 736 * 8 * 8 = 47k params per filter? No, params are fine.
        # But computation is high. Let's see if we can reduce.
        # The classes are fixed. We can't change their internal channel counts easily without editing them.
        # BUT, we can edit the classes too! The prompt says "change only the architecture".
        # I will edit the Inception classes to have smaller channel counts suitable for CIFAR-
        # 10 and 1-epoch training.
        
        # Let's redefine the classes to be lighter.
        # I will replace the entire file content to include lighter Inception blocks.
        
        self.Mixed_6a = inception_b(256)
        # With my new lightweight InceptionB, let's say it outputs 192 channels.
        # Let's design the new blocks:
        # InceptionA_Light: in->out. 
        # Branch 1x1: 32
        # Branch 5x5: 16->32
        # Branch 3x3dbl: 32->48->48
        # Branch Pool: 16
        # Total: 32+32+48+16 = 128.
        
        # InceptionB_Light: in->out (stride 2)
        # Branch 3x3: 96
        # Branch 3x3dbl: 32->48->48
        # Branch Pool: in
        # Total: 96+48+in.
        
        # InceptionC_Light: in->out
        # Branch 1x1: 64
        # Branch 7x7: 32->32->64
        # Branch 7x7dbl: 32->32->32->32->64
        # Branch Pool: 64
        # Total: 64+64+64+64 = 256.
        
        # InceptionD_Light: in->out (stride 2)
        # Branch 3x3: 64->128
        # Branch 7x7x3: 64->64->64->128
        # Branch Pool: in
        # Total: 128+128+in.
        
        # InceptionE_Light: in->out
        # Branch 1x1: 128
        # Branch 3x3: 128->128+128
        # Branch 3x3dbl: 128->128->128+128
        # Branch Pool: 64
        # Total: 128+256+256+64 = 704.
        
        # This is getting complicated to guess exact dims without seeing the code.
        # I will rewrite the classes to be simple, efficient ResNet-like or simple Inception-lite.
        # Actually, the simplest and most effective for CIFAR-10 in 1 epoch is a ResNet-like structure.
        # But the prompt asks to return edits to the *given* file.
        # I will replace the Net class and the Inception classes with a lighter, more CIFAR-friendly architecture.
        # I will keep the class names to minimize confusion but change their internals to be "Light" versions.
        
        # Transition 6a: 144 input.
        # InceptionB outputs: 96 (3x3 stride 2) + 48 (3x3dbl stride 2) + 144 (pool stride 2) = 288.
        self.Mixed_6a = inception_b(144)
        
        # Block 2: 4x4 input (after stride 2 in 6a)
        self.Mixed_6b = inception_c(288, channels_7x7=32)
        # Output: 64 + 64 + 64 + 64 = 256
        
        self.Mixed_6c = inception_c(256, channels_7x7=32)
        # Output: 256
        
        self.Mixed_6d = inception_c(256, channels_7x7=32)
        # Output: 256
        
        self.Mixed_6e = inception_c(256, channels_7x7=32)
        # Output: 256
        
        # Transition 7a: 256 input.
        # InceptionD outputs: 128 (3x3 stride 2) + 128 (7x7x3 stride 2) + 256 (pool stride 2) = 512.
        self.Mixed_7a = inception_d(256)
        
        # Block 3: 2x2 input (after stride 2 in 7a)
        self.Mixed_7b = inception_e(512)
        # Output: 128 + 256 + 256 + 64 = 704
        
        self.Mixed_7c = inception_e(704)
        # Output: 704
        
        self.AuxLogits = None
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(704, num_classes)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]) -> InceptionOutputs:
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._transform_input(x)
        x, aux = self._forward(x)
        return x


class InceptionA(nn.Module):
    def __init__(
            self, in_channels: int, pool_features: int, conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        # Lighter channels for CIFAR-10
        self.branch1x1 = conv_block(in_channels, 32, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = conv_block(16, 32, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(32, 48, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(48, 48, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        # Lighter channels
        self.branch3x3 = conv_block(in_channels, 96, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(32, 48, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(48, 48, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(
            self, in_channels: int, channels_7x7: int, conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        # Lighter channels
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 64, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 64, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, 64, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        # Lighter channels
        self.branch3x3_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = conv_block(64, 128, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch7x7x3_2 = conv_block(64, 64, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(64, 64, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(64, 128, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        # Lighter channels
        self.branch1x1 = conv_block(in_channels, 128, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 128, kernel_size=1)
        self.branch3x3_2a = conv_block(128, 128, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(128, 128, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 128, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(128, 128, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(128, 128, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(128, 128, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 64, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(
            self, in_channels: int, num_classes: int, conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x: Tensor) -> Tensor:
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)