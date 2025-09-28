# Auto-generated single-file for RadiusInteractionGraph
# Dependencies are emitted in topological order (utilities first).

# Standard library and external imports
import torch
from torch import Tensor
from typing import Optional, Tuple
import warnings
from torch import Tensor as OptTensor
from typing import Tuple
class torch_geometric: 
    class typing:
        WITH_TORCH_CLUSTER_BATCH_SIZE = False
class torch_cluster:
    @staticmethod
    def radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32, flow='source_to_target', num_workers=1, batch_size=None):
        B = x.shape[0] if batch is None else batch.max().item() + 1
        N = x.shape[0]
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
        return edge_index
from typing import Optional
from torch import nn

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

# ---- RadiusInteractionGraph (target) ----
class RadiusInteractionGraph(torch.nn.Module):
    r"""Creates edges based on atom positions :obj:`pos` to all points within
    the cutoff distance.

    Args:
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance with the
            default interaction graph method.
            (default: :obj:`32`)
    """
    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Forward pass.

        Args:
            pos (Tensor): Coordinates of each atom.
            batch (LongTensor, optional): Batch indices assigning each atom to
                a separate molecule.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight

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
        self.radius_graph = RadiusInteractionGraph(cutoff=5.0, max_num_neighbors=16)
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        
        pos = torch.rand(B * H * W, 2, device=self.device)
        batch = torch.arange(B, device=self.device).repeat_interleave(H * W)
        
        edge_index, edge_weight = self.radius_graph(pos, batch)
        
        x = x.mean(dim=(2, 3))
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, data_roll):
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
