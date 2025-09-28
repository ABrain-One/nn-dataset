# Auto-generated single-file for FCOSRegressionHead
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable
from functools import partial
from typing import Callable
from typing import Optional

# ---- original imports from contributing modules ----
from torch import nn, Tensor
from typing import Callable, Optional

# ---- FCOSRegressionHead (target) ----
class FCOSRegressionHead(nn.Module):
    """
    A regression head for use in FCOS, which combines regression branch and center-ness branch.
    This can obtain better performance.

    Reference: `FCOS: A simple and strong anchor-free object detector <https://arxiv.org/abs/2006.09214>`_.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 4.
        norm_layer: Module specifying the normalization layer to use.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_convs: int = 4,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(nn.GroupNorm, 32)

        conv = []
        for _ in range(num_convs):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(norm_layer(in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        self.bbox_ctrness = nn.Conv2d(in_channels, num_anchors * 1, kernel_size=3, stride=1, padding=1)
        for layer in [self.bbox_reg, self.bbox_ctrness]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.zeros_(layer.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x: list[Tensor]) -> tuple[Tensor, Tensor]:
        all_bbox_regression = []
        all_bbox_ctrness = []

        for features in x:
            bbox_feature = self.conv(features)
            bbox_regression = nn.functional.relu(self.bbox_reg(bbox_feature))
            bbox_ctrness = self.bbox_ctrness(bbox_feature)

            # permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)
            all_bbox_regression.append(bbox_regression)

            # permute bbox ctrness output from (N, 1 * A, H, W) to (N, HWA, 1).
            bbox_ctrness = bbox_ctrness.view(N, -1, 1, H, W)
            bbox_ctrness = bbox_ctrness.permute(0, 3, 4, 1, 2)
            bbox_ctrness = bbox_ctrness.reshape(N, -1, 1)
            all_bbox_ctrness.append(bbox_ctrness)

        return torch.cat(all_bbox_regression, dim=1), torch.cat(all_bbox_ctrness, dim=1)

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
        self.fcos_reg_head = FCOSRegressionHead(
            in_channels=32,
            num_anchors=1,
            num_convs=2
        )
        self.classifier = nn.Linear(32, self.num_classes)

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
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
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
