# Auto-generated single-file for HeteroLayerNorm
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# e, s
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
from typing import List, Optional, Union
from torch import Tensor as OptTensor
from typing import List
from typing import Union
from typing import Optional

# ---- original imports from contributing modules ----
from torch.nn import Parameter

# ---- HeteroLayerNorm (target) ----
class HeteroLayerNorm(torch.nn.Module):
    r"""Applies layer normalization over each individual example in a batch
    of heterogeneous features as described in the `"Layer Normalization"
    <https://arxiv.org/abs/1607.06450>`_ paper.
    Compared to :class:`LayerNorm`, :class:`HeteroLayerNorm` applies
    normalization individually for each node or edge type.

    Args:
        in_channels (int): Size of each input sample.
        num_types (int): The number of types.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        mode (str, optional): The normalization mode to use for layer
            normalization (:obj:`"node"`). If `"node"` is used, each node will
            be considered as an element to be normalized.
            (default: :obj:`"node"`)
        device (torch.device, optional): The device to use for the module.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        in_channels: int,
        num_types: int,
        eps: float = 1e-5,
        affine: bool = True,
        mode: str = 'node',
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert mode == 'node'

        self.in_channels = in_channels
        self.num_types = num_types
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = Parameter(
                torch.empty(num_types, in_channels, device=device))
            self.bias = Parameter(
                torch.empty(num_types, in_channels, device=device))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Tensor,
        type_vec: OptTensor = None,
        type_ptr: Optional[Union[Tensor, List[int]]] = None,
    ) -> Tensor:
        r"""Forward pass.

        .. note::
            Either :obj:`type_vec` or :obj:`type_ptr` needs to be specified.
            In general, relying on :obj:`type_ptr` is more efficient in case
            the input tensor is sorted by types.

        Args:
            x (torch.Tensor): The input features.
            type_vec (torch.Tensor, optional): A vector that maps each entry to
                a type. (default: :obj:`None`)
            type_ptr (torch.Tensor or List[int]): A vector denoting the
                boundaries of types. (default: :obj:`None`)
        """
        if type_vec is None and type_ptr is None:
            raise ValueError("Either 'type_vec' or 'type_ptr' must be given")

        out = F.layer_norm(x, (self.in_channels, ), None, None, self.eps)

        if self.affine:
            # TODO Revisit this logic completely as it performs worse than just
            # operating on a dictionary of tensors
            # (especially the `type_vec` code path)
            if type_ptr is not None:
                h = torch.empty_like(out)
                for i, (s, e) in enumerate(zip(type_ptr[:-1], type_ptr[1:])):
                    h[s:e] = out[s:e] * self.weight[i] + self.bias[i]
                out = h
            else:
                out = out * self.weight[type_vec] + self.bias[type_vec]

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'num_types={self.num_types})')

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
        self.hetero_layer_norm = HeteroLayerNorm(in_channels=32, num_types=1, eps=1e-5, affine=True, mode='node', device=device)
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
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        type_vec = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.hetero_layer_norm(x, type_vec=type_vec)
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
