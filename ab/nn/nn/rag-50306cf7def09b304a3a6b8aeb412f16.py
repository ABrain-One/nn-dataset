# Auto-generated single-file for HighResolutionModule
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# __name__
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
from typing import List
import logging

# ---- timm.models.hrnet._BN_MOMENTUM ----
_BN_MOMENTUM = 0.1

# ---- timm.models.hrnet._logger ----
_logger = logging.getLogger(__name__)

# ---- BasicBlock for HighResolutionModule ----
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=_BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# ---- HighResolutionModule (target) ----
class HighResolutionModule(nn.Module):
    def __init__(
            self,
            num_branches,
            block_types,
            num_blocks,
            num_in_chs,
            num_channels,
            fuse_method,
            multi_scale_output=True,
    ):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches,
            block_types,
            num_blocks,
            num_in_chs,
            num_channels,
        )

        self.num_in_chs = num_in_chs
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches,
            block_types,
            num_blocks,
            num_channels,
        )
        self.fuse_layers = self._make_fuse_layers()
        self.fuse_act = nn.ReLU(False)

    def _check_branches(self, num_branches, block_types, num_blocks, num_in_chs, num_channels):
        error_msg = ''
        if num_branches != len(num_blocks):
            error_msg = 'num_branches({}) <> num_blocks({})'.format(num_branches, len(num_blocks))
        elif num_branches != len(num_channels):
            error_msg = 'num_branches({}) <> num_channels({})'.format(num_branches, len(num_channels))
        elif num_branches != len(num_in_chs):
            error_msg = 'num_branches({}) <> num_in_chs({})'.format(num_branches, len(num_in_chs))
        if error_msg:
            _logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block_type, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_in_chs[branch_index] != num_channels[branch_index] * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_in_chs[branch_index], num_channels[branch_index] * block_type.expansion,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block_type.expansion, momentum=_BN_MOMENTUM),
            )

        layers = [block_type(self.num_in_chs[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_in_chs[branch_index] = num_channels[branch_index] * block_type.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block_type(self.num_in_chs[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block_type, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block_type, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return nn.Identity()

        num_branches = self.num_branches
        num_in_chs = self.num_in_chs
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_in_chs[j], num_in_chs[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_in_chs[i], momentum=_BN_MOMENTUM),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_out_chs_conv3x3 = num_in_chs[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_in_chs[j], num_out_chs_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_out_chs_conv3x3, momentum=_BN_MOMENTUM)
                            ))
                        else:
                            num_out_chs_conv3x3 = num_in_chs[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_in_chs[j], num_out_chs_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_out_chs_conv3x3, momentum=_BN_MOMENTUM),
                                nn.ReLU(False)
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_in_chs(self):
        return self.num_in_chs

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i, branch in enumerate(self.branches):
            x[i] = branch(x[i])

        x_fuse = []
        for i, fuse_outer in enumerate(self.fuse_layers):
            y = None
            for j, f in enumerate(fuse_outer):
                if y is None:
                    y = f(x[j])
                else:
                    y = y + f(x[j])
            x_fuse.append(self.fuse_act(y))
        return x_fuse

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.high_resolution_module = HighResolutionModule(
            num_branches=2,
            block_types=BasicBlock,
            num_blocks=[1, 1],
            num_in_chs=[32, 32],
            num_channels=[32, 32],
            fuse_method='sum',
            multi_scale_output=True
        )
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x_downsampled = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        x_list = [x, x_downsampled]
        x_fused = self.high_resolution_module(x_list)
        # Use only the first fused output to avoid dimension mismatch
        x = x_fused[0]
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
