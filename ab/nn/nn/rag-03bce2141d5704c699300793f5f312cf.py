# Auto-generated single-file for DropBlock2d
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any
from types import FunctionType

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

# ---- torchvision.ops.drop_block.drop_block2d ----
def drop_block2d(
    input: Tensor, p: float, block_size: int, inplace: bool = False, eps: float = 1e-06, training: bool = True
) -> Tensor:
    """
    Implements DropBlock2d from `"DropBlock: A regularization method for convolutional networks"
    <https://arxiv.org/abs/1810.12890>`.

    Args:
        input (Tensor[N, C, H, W]): The input tensor or 4-dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): Probability of an element to be dropped.
        block_size (int): Size of the block to drop.
        inplace (bool): If set to ``True``, will do this operation in-place. Default: ``False``.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-6.
        training (bool): apply dropblock if is ``True``. Default: ``True``.

    Returns:
        Tensor[N, C, H, W]: The randomly zeroed tensor after dropblock.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(drop_block2d)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}.")
    if input.ndim != 4:
        raise ValueError(f"input should be 4 dimensional. Got {input.ndim} dimensions.")
    if not training or p == 0.0:
        return input

    N, C, H, W = input.size()
    block_size = min(block_size, W, H)
    if block_size % 2 == 0:
        raise ValueError(f"block size should be odd. Got {block_size} which is even.")

    # compute the gamma of Bernoulli distribution
    gamma = (p * H * W) / ((block_size**2) * ((H - block_size + 1) * (W - block_size + 1)))
    noise = torch.empty((N, C, H - block_size + 1, W - block_size + 1), dtype=input.dtype, device=input.device)
    noise.bernoulli_(gamma)

    noise = F.pad(noise, [block_size // 2] * 4, value=0)
    noise = F.max_pool2d(noise, stride=(1, 1), kernel_size=(block_size, block_size), padding=block_size // 2)
    noise = 1 - noise
    normalize_scale = noise.numel() / (eps + noise.sum())
    if inplace:
        input.mul_(noise).mul_(normalize_scale)
    else:
        input = input * noise * normalize_scale
    return input

# ---- DropBlock2d (target) ----
class DropBlock2d(nn.Module):
    """
    See :func:`drop_block2d`.
    """

    def __init__(self, p: float, block_size: int, inplace: bool = False, eps: float = 1e-06) -> None:
        super().__init__()

        self.p = p
        self.block_size = block_size
        self.inplace = inplace
        self.eps = eps

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Input feature map on which some areas will be randomly
                dropped.
        Returns:
            Tensor: The tensor after DropBlock layer.
        """
        return drop_block2d(input, self.p, self.block_size, self.inplace, self.eps, self.training)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, block_size={self.block_size}, inplace={self.inplace})"
        return s

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
        
        self.dropblock2d = DropBlock2d(
            p=0.1,
            block_size=3,
            inplace=False,
            eps=1e-06
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
        x = self.dropblock2d(x)
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
