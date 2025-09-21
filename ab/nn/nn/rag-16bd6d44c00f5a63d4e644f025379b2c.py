# Auto-generated single-file for Distance
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# torch_cluster
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import warnings
from torch import Tensor as OptTensor
from typing import Tuple
class torch_geometric: 
    class typing:
        WITH_TORCH_CLUSTER_BATCH_SIZE = True

class torch_cluster:
    @staticmethod
    def radius_graph(x, r, batch, loop, max_num_neighbors, flow, num_workers, batch_size=None):
        # Simple fallback implementation
        N = x.size(0)
        edge_list = []
        for i in range(N):
            for j in range(N):
                if i != j or loop:
                    dist = torch.norm(x[i] - x[j])
                    if dist <= r:
                        edge_list.append([i, j])
        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long, device=x.device)
        return torch.tensor(edge_list, dtype=torch.long, device=x.device).t()

# ---- torch_geometric.nn.pool.__init__.radius_graph ----
def radius_graph(
    x: Tensor,
    r: float,
    batch: OptTensor = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = 'source_to_target',
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Computes graph edges to all points within a given distance.

    .. code-block:: python

        import torch
        from torch_geometric.nn import radius_graph

        x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)
        flow (str, optional): The flow direction when using in combination with
            message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor`

    .. warning::

        The CPU implementation of :meth:`radius_graph` with
        :obj:`max_num_neighbors` is biased towards certain quadrants.
        Consider setting :obj:`max_num_neighbors` to :obj:`None` or moving
        inputs to GPU before proceeding.
    """
    if batch is not None and x.device != batch.device:
        warnings.warn(
            "Input tensor 'x' and 'batch' are on different devices "
            "in 'radius_graph'. Performing blocking device transfer",
            stacklevel=2)
        batch = batch.to(x.device)

    if not torch_geometric.typing.WITH_TORCH_CLUSTER_BATCH_SIZE:
        return torch_cluster.radius_graph(x, r, batch, loop, max_num_neighbors,
                                          flow, num_workers)
    return torch_cluster.radius_graph(x, r, batch, loop, max_num_neighbors,
                                      flow, num_workers, batch_size)

# ---- Distance (target) ----
class Distance(torch.nn.Module):
    r"""Computes the pairwise distances between atoms in a molecule.

    This module computes the pairwise distances between atoms in a molecule,
    represented by their positions :obj:`pos`.
    The distances are computed only between points that are within a certain
    cutoff radius.

    Args:
        cutoff (float): The cutoff radius beyond
            which distances are not computed.
        max_num_neighbors (int, optional): The maximum number of neighbors
            considered for each point. (default: :obj:`32`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not
            include self-loops. (default: :obj:`True`)
    """
    def __init__(
        self,
        cutoff: float,
        max_num_neighbors: int = 32,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.add_self_loops = add_self_loops

    def forward(
        self,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Computes the pairwise distances between atoms in the molecule.

        Args:
            pos (torch.Tensor): The positions of the atoms in the molecule.
            batch (torch.Tensor): A batch vector, which assigns each node to a
                specific example.

        Returns:
            edge_index (torch.Tensor): The indices of the edges in the graph.
            edge_weight (torch.Tensor): The distances between connected nodes.
            edge_vec (torch.Tensor): The vector differences between connected
                nodes.
        """
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            loop=self.add_self_loops,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.add_self_loops:
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        return edge_index, edge_weight, edge_vec

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
        
        self.distance = Distance(
            cutoff=1.0,
            max_num_neighbors=32,
            add_self_loops=True
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
