# Auto-generated single-file for Select
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
from torch import Tensor
from typing import Optional
from dataclasses import dataclass

# ---- torch_geometric.nn.pool.select.base.SelectOutput ----
class SelectOutput:
    r"""The output of the :class:`Select` method, which holds an assignment
    from selected nodes to their respective cluster(s).

    Args:
        node_index (torch.Tensor): The indices of the selected nodes.
        num_nodes (int): The number of nodes.
        cluster_index (torch.Tensor): The indices of the clusters each node in
            :obj:`node_index` is assigned to.
        num_clusters (int): The number of clusters.
        weight (torch.Tensor, optional): A weight vector, denoting the strength
            of the assignment of a node to its cluster. (default: :obj:`None`)
    """
    node_index: Tensor
    num_nodes: int
    cluster_index: Tensor
    num_clusters: int
    weight: Optional[Tensor] = None

    def __init__(
        self,
        node_index: Tensor,
        num_nodes: int,
        cluster_index: Tensor,
        num_clusters: int,
        weight: Optional[Tensor] = None,
    ):
        if node_index.dim() != 1:
            raise ValueError(f"Expected 'node_index' to be one-dimensional "
                             f"(got {node_index.dim()} dimensions)")

        if cluster_index.dim() != 1:
            raise ValueError(f"Expected 'cluster_index' to be one-dimensional "
                             f"(got {cluster_index.dim()} dimensions)")

        if node_index.numel() != cluster_index.numel():
            raise ValueError(f"Expected 'node_index' and 'cluster_index' to "
                             f"hold the same number of values (got "
                             f"{node_index.numel()} and "
                             f"{cluster_index.numel()} values)")

        if weight is not None and weight.dim() != 1:
            raise ValueError(f"Expected 'weight' vector to be one-dimensional "
                             f"(got {weight.dim()} dimensions)")

        if weight is not None and weight.numel() != node_index.numel():
            raise ValueError(f"Expected 'weight' to hold {node_index.numel()} "
                             f"values (got {weight.numel()} values)")

        self.node_index = node_index
        self.num_nodes = num_nodes
        self.cluster_index = cluster_index
        self.num_clusters = num_clusters
        self.weight = weight

# ---- Select (target) ----
class Select(torch.nn.Module):
    r"""An abstract base class for implementing custom node selections as
    described in the `"Understanding Pooling in Graph Neural Networks"
    <https://arxiv.org/abs/1905.05178>`_ paper, which maps the nodes of an
    input graph to supernodes in the coarsened graph.

    Specifically, :class:`Select` returns a :class:`SelectOutput` output, which
    holds a (sparse) mapping :math:`\mathbf{C} \in {[0, 1]}^{N \times C}` that
    assigns selected nodes to one or more of :math:`C` super nodes.
    """
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""

    def forward(self, *args, **kwargs) -> SelectOutput:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class ConcreteSelect(Select):
    def __init__(self, num_nodes, num_clusters):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        
    def forward(self, x):
        batch_size = x.size(0)
        node_indices = torch.arange(self.num_nodes, device=x.device).unsqueeze(0).repeat(batch_size, 1).flatten()
        cluster_indices = torch.randint(0, self.num_clusters, (batch_size * self.num_nodes,), device=x.device)
        return SelectOutput(
            node_index=node_indices,
            num_nodes=self.num_nodes,
            cluster_index=cluster_indices,
            num_clusters=self.num_clusters
        )

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
        self.select = ConcreteSelect(num_nodes=64, num_clusters=32)
        self.classifier = torch.nn.Linear(64, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W)
        select_output = self.select(x_flat)
        x = x_flat.mean(dim=2)
        x = torch.nn.functional.adaptive_avg_pool2d(x.unsqueeze(-1).unsqueeze(-1), (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = torch.nn.CrossEntropyLoss()

    def learn(self, data_roll):
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
