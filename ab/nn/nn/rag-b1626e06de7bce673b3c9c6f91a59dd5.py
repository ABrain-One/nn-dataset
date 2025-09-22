# Auto-generated single-file for Connect
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
from torch import Tensor
from typing import Optional
from dataclasses import dataclass

# ---- torch_geometric.nn.pool.connect.base.ConnectOutput ----
class ConnectOutput:
    r"""The output of the :class:`Connect` method, which holds the coarsened
    graph structure, and optional pooled edge features and batch vectors.

    Args:
        edge_index (torch.Tensor): The edge indices of the cooarsened graph.
        edge_attr (torch.Tensor, optional): The pooled edge features of the
            coarsened graph. (default: :obj:`None`)
        batch (torch.Tensor, optional): The pooled batch vector of the
            coarsened graph. (default: :obj:`None`)
    """
    edge_index: Tensor
    edge_attr: Optional[Tensor] = None
    batch: Optional[Tensor] = None

    def __init__(
        self,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ):
        if edge_index.dim() != 2:
            raise ValueError(f"Expected 'edge_index' to be two-dimensional "
                             f"(got {edge_index.dim()} dimensions)")

        if edge_index.size(0) != 2:
            raise ValueError(f"Expected 'edge_index' to have size '2' in the "
                             f"first dimension (got '{edge_index.size(0)}')")

        if edge_attr is not None and edge_attr.size(0) != edge_index.size(1):
            raise ValueError(f"Expected 'edge_index' and 'edge_attr' to "
                             f"hold the same number of edges (got "
                             f"{edge_index.size(1)} and {edge_attr.size(0)} "
                             f"edges)")

        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.batch = batch

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

# ---- Connect (target) ----
class Connect(torch.nn.Module):
    r"""An abstract base class for implementing custom edge connection
    operators as described in the `"Understanding Pooling in Graph Neural
    Networks" <https://arxiv.org/abs/1905.05178>`_ paper.

    Specifically, :class:`Connect` determines for each pair of supernodes the
    presence or abscene of an edge based on the existing edges between the
    nodes in the two supernodes.
    The operator also computes pooled edge features and batch vectors
    (if present).
    """
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""

    def forward(
        self,
        select_output: SelectOutput,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> ConnectOutput:
        r"""Forward pass.

        Args:
            select_output (SelectOutput): The output of :class:`Select`.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific graph. (default: :obj:`None`)
        """
        raise NotImplementedError

    def get_pooled_batch(
        select_output: SelectOutput,
        batch: Optional[Tensor],
    ) -> Optional[Tensor]:
        r"""Returns the batch vector of the coarsened graph.

        Args:
            select_output (SelectOutput): The output of :class:`Select`.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example. (default: :obj:`None`)
        """
        if batch is None:
            return batch

        out = torch.arange(select_output.num_clusters, device=batch.device)
        return out.scatter_(0, select_output.cluster_index,
                            batch[select_output.node_index])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

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
        self.connect = Connect()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(32, self.num_classes)

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

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
