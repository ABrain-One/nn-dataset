# Auto-generated single-file for GraphSizeNorm
# Dependencies are emitted in topological order (utilities first).

# Standard library and external imports
import torch
from torch import Tensor
from typing import Optional, Tuple, Union
from torch import Tensor as OptTensor
class SparseTensor: pass
class torch_geometric: pass
class EdgeIndex:
    def get_sparse_size(self): return (0, 0)
from typing import Union
from typing import Tuple
from typing import Optional

# ---- torch_geometric.utils.num_nodes.maybe_num_nodes ----
def maybe_num_nodes(
    edge_index: Union[Tensor, Tuple[Tensor, Tensor], SparseTensor],
    num_nodes: Optional[int] = None,
) -> int:
    if num_nodes is not None:
        return num_nodes
    elif not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
        return max(edge_index.get_sparse_size())
    elif isinstance(edge_index, Tensor):
        if torch_geometric.utils.is_torch_sparse_tensor(edge_index):
            return max(edge_index.size(0), edge_index.size(1))

        if torch.jit.is_tracing():
            # Avoid non-traceable if-check for empty `edge_index` tensor:
            tmp = torch.concat([
                edge_index.view(-1),
                edge_index.new_full((1, ), fill_value=-1)
            ])
            return tmp.max() + 1  # type: ignore

        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    elif isinstance(edge_index, tuple):
        return max(
            int(edge_index[0].max()) + 1 if edge_index[0].numel() > 0 else 0,
            int(edge_index[1].max()) + 1 if edge_index[1].numel() > 0 else 0,
        )
    elif isinstance(edge_index, SparseTensor):
        return max(edge_index.size(0), edge_index.size(1))
    raise NotImplementedError

# ---- torch_geometric.utils._degree.degree ----
def degree(index: Tensor, num_nodes: Optional[int] = None,
           dtype: Optional[torch.dtype] = None) -> Tensor:
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`

    Example:
        >>> row = torch.tensor([0, 1, 0, 2, 0])
        >>> degree(row, dtype=torch.long)
        tensor([3, 1, 1])
    """
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N, ), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)

# ---- GraphSizeNorm (target) ----
class GraphSizeNorm(torch.nn.Module):
    r"""Applies Graph Size Normalization over each individual graph in a batch
    of node features as described in the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x}_i}{\sqrt{|\mathcal{V}|}}
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, batch: OptTensor = None,
                batch_size: Optional[int] = None) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The source tensor.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example. (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given. (default: :obj:`None`)
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            batch_size = 1

        inv_sqrt_deg = degree(batch, batch_size, dtype=x.dtype).pow(-0.5)
        return x * inv_sqrt_deg.index_select(0, batch).view(-1, 1)

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
        self.graph_size_norm = GraphSizeNorm()
        self.classifier = torch.nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=False),
        ]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2)
        x_flat = x_flat.contiguous().view(-1, C)
        batch = torch.arange(B, device=x.device).repeat_interleave(H * W)
        x_norm = self.graph_size_norm(x_flat, batch, B)
        x_norm = x_norm.view(B, H * W, C).transpose(1, 2).contiguous().view(B, C, H, W)
        x = x + x_norm
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
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
