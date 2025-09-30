# Auto-generated single-file for PatchEmbed3d
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Optional, Callable
from types import FunctionType
from typing import Any
from typing import Optional
from typing import Callable

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

# ---- torchvision.models.video.swin_transformer._compute_pad_size_3d ----
def _compute_pad_size_3d(size_dhw: tuple[int, int, int], patch_size: tuple[int, int, int]) -> tuple[int, int, int]:
    pad_size = [(patch_size[i] - size_dhw[i] % patch_size[i]) % patch_size[i] for i in range(3)]
    return pad_size[0], pad_size[1], pad_size[2]

# ---- PatchEmbed3d (target) ----
class PatchEmbed3d(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (List[int]): Patch token size.
        in_channels (int): Number of input channels. Default: 3
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size: list[int],
        in_channels: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.tuple_patch_size = (patch_size[0], patch_size[1], patch_size[2])

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.tuple_patch_size,
            stride=self.tuple_patch_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        # padding
        _, _, t, h, w = x.size()
        pad_size = _compute_pad_size_3d((t, h, w), self.tuple_patch_size)
        x = F.pad(x, (0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
        x = self.proj(x)  # B C T Wh Ww
        x = x.permute(0, 2, 3, 4, 1)  # B T Wh Ww C
        if self.norm is not None:
            x = self.norm(x)
        return x

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        self.patch_embed_3d = PatchEmbed3d(
            patch_size=[1, 4, 4],
            in_channels=32,
            embed_dim=64,
            norm_layer=nn.LayerNorm
        )
        
        self.classifier = nn.Linear(64, self.num_classes)
        
    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.unsqueeze(2)
        x = self.patch_embed_3d(x)
        x = x.mean(dim=(1, 2, 3))
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)
    
    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)
