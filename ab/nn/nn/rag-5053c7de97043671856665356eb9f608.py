# Auto-generated single-file for DenseGraphConv
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# Linear
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
from torch import Tensor
from typing import Optional

# ---- original imports from contributing modules ----
from torch.nn import Linear

# ---- DenseGraphConv (target) ----
class DenseGraphConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GraphConv`."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'add',
        bias: bool = True,
    ):
        assert aggr in ['add', 'mean', 'max']
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

        self.lin_rel = Linear(in_channels, out_channels, bias=bias)
        self.lin_root = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x: Tensor, adj: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
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
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, C = x.size()

        if self.aggr == 'add':
            out = torch.matmul(adj, x)
        elif self.aggr == 'mean':
            out = torch.matmul(adj, x)
            out = out / adj.sum(dim=-1, keepdim=True).clamp_(min=1)
        elif self.aggr == 'max':
            out = x.unsqueeze(-2).repeat(1, 1, N, 1)
            adj = adj.unsqueeze(-1).expand(B, N, N, C)
            out[adj == 0] = float('-inf')
            out = out.max(dim=-3)[0]
            out[out == float('-inf')] = 0.
        else:
            raise NotImplementedError

        out = self.lin_rel(out)
        out = out + self.lin_root(x)

        if mask is not None:
            out = out * mask.view(-1, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')

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
        
        self.dense_graph_conv = DenseGraphConv(
            in_channels=32,
            out_channels=32,
            aggr='add',
            bias=True
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
