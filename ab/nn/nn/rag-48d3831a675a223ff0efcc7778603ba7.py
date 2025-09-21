# Auto-generated single-file for DenseGINConv
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# Module
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
from typing import Any, Optional
from typing import Any
from typing import Optional

# ---- torch_geometric.nn.inits.reset ----
def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

# ---- DenseGINConv (target) ----
class DenseGINConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GINConv`."""
    def __init__(
        self,
        nn: Module,
        eps: float = 0.0,
        train_eps: bool = False,
    ):
        super().__init__()

        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None,
                add_loop: bool = True) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
                The adjacency tensor is broadcastable in the batch dimension,
                resulting in a shared adjacency matrix for the complete batch.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        out = torch.matmul(adj, x)
        if add_loop:
            out = (1 + self.eps) * x + out

        out = self.nn(out)

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(torch.nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(32, self.num_classes)
        
        gin_nn = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32)
        )
        self.dense_gin = DenseGINConv(
            nn=gin_nn,
            eps=0.0,
            train_eps=True
        )

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
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
