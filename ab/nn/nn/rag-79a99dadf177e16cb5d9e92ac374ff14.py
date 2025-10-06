# Auto-generated single-file for DlaTree
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import List, Optional
from typing import List
from typing import Optional

# ---- timm.models.dla.DlaRoot ----
class DlaRoot(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut):
        super(DlaRoot, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut

    def forward(self, x_children: List[torch.Tensor]):
        x = self.conv(torch.cat(x_children, 1))
        x = self.bn(x)
        if self.shortcut:
            x += x_children[0]
        x = self.relu(x)

        return x

# ---- DlaTree (target) ----
class DlaTree(nn.Module):
    def __init__(
            self,
            levels,
            block,
            in_channels,
            out_channels,
            stride=1,
            dilation=1,
            cardinality=1,
            base_width=64,
            level_root=False,
            root_dim=0,
            root_kernel_size=1,
            root_shortcut=False,
    ):
        super(DlaTree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        self.downsample = nn.MaxPool2d(stride, stride=stride) if stride > 1 else nn.Identity()
        self.project = nn.Identity()
        cargs = dict(dilation=dilation, cardinality=cardinality, base_width=base_width)
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, **cargs)
            self.tree2 = block(out_channels, out_channels, 1, **cargs)
            if in_channels != out_channels:
                # NOTE the official impl/weights have  project layers in levels > 1 case that are never
                # used, I've moved the project layer here to avoid wasted params but old checkpoints will
                # need strict=False while loading.
                self.project = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_channels))
            self.root = DlaRoot(root_dim, out_channels, root_kernel_size, root_shortcut)
        else:
            cargs.update(dict(root_kernel_size=root_kernel_size, root_shortcut=root_shortcut))
            self.tree1 = DlaTree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                **cargs,
            )
            self.tree2 = DlaTree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                **cargs,
            )
            self.root = None
        self.level_root = level_root
        self.root_dim = root_dim
        self.levels = levels

    def forward(self, x, shortcut: Optional[torch.Tensor] = None, children: Optional[List[torch.Tensor]] = None):
        if children is None:
            children = []
        bottom = self.downsample(x)
        shortcut = self.project(bottom)
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, shortcut)
        if self.root is not None:  # levels == 1
            x2 = self.tree2(x1)
            x = self.root([x2, x1] + children)
        else:
            children.append(x1)
            x = self.tree2(x1, None, children)
        return x

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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, self.num_classes)
        
        from torch.nn import BatchNorm2d, ReLU
        class SimpleBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1, **kwargs):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
                self.bn1 = BatchNorm2d(out_channels)
                self.relu = ReLU(inplace=True)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
                self.bn2 = BatchNorm2d(out_channels)
                self.shortcut = nn.Sequential() if in_channels == out_channels else nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    BatchNorm2d(out_channels)
                )
            
            def forward(self, x, shortcut=None, children=None):
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                out += self.shortcut(x)
                out = self.relu(out)
                return out
        
        self.dla_tree = DlaTree(
            levels=2,
            block=SimpleBlock,
            in_channels=32,
            out_channels=32,
            stride=1,
            dilation=1,
            cardinality=1,
            base_width=64,
            level_root=False,
            root_dim=0,
            root_kernel_size=1,
            root_shortcut=False
        )

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
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
