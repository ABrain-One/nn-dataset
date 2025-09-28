# Auto-generated single-file for LRASPP
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any
from collections import OrderedDict
from types import FunctionType

# ---- torchvision.models.segmentation.lraspp.LRASPPHead ----
class LRASPPHead(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, num_classes: int, inter_channels: int) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, input: dict[str, Tensor]) -> Tensor:
        low = input["low"]
        high = input["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        return self.low_classifier(low) + self.high_classifier(x)

# ---- torchvision.utils._log_api_usage_once ----
def _log_api_usage_once(obj: Any) -> None:
    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;

    Args:
        obj (class instance or method): an object to extract info from.
    """
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")

# ---- LRASPP (target) ----
class LRASPP(nn.Module):
    """
    Implements a Lite R-ASPP Network for semantic segmentation from
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "high" for the high level feature map and "low" for the low level feature map.
        low_channels (int): the number of channels of the low level features.
        high_channels (int): the number of channels of the high level features.
        num_classes (int, optional): number of output classes of the model (including the background).
        inter_channels (int, optional): the number of channels for intermediate computations.
    """

    def __init__(
        self, backbone: nn.Module, low_channels: int, high_channels: int, num_classes: int, inter_channels: int = 128
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.backbone = backbone
        self.classifier = LRASPPHead(low_channels, high_channels, num_classes, inter_channels)

    def forward(self, input: Tensor) -> dict[str, Tensor]:
        features = self.backbone(input)
        out = self.classifier(features)
        out = F.interpolate(out, size=input.shape[-2:], mode="bilinear", align_corners=False)

        result = OrderedDict()
        result["out"] = out

        return result

class SimpleBackbone(nn.Module):
    def __init__(self, in_channels, low_channels, high_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, low_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(low_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(low_channels, high_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(high_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        low = self.relu1(self.bn1(self.conv1(x)))
        high = self.relu2(self.bn2(self.conv2(low)))
        return {"low": low, "high": high}

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
        self.backbone = SimpleBackbone(self.in_channels, 16, 32)
        self.lraspp = LRASPP(self.backbone, low_channels=16, high_channels=32, num_classes=self.num_classes, inter_channels=16)
        self.classifier = nn.Linear(self.num_classes, self.num_classes)

    def forward(self, x):
        result = self.lraspp(x)
        out = result["out"]
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.flatten(1)
        return self.classifier(out)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self.optimizer.step()
