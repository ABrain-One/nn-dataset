# Auto-generated single-file for InducedSetAttentionBlock
# Dependencies are emitted in topological order (utilities first).

# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
from typing import Optional
from torch.nn import LayerNorm, Linear, MultiheadAttention

# ---- torch_geometric.nn.aggr.utils.MultiheadAttentionBlock ----
class MultiheadAttentionBlock(torch.nn.Module):
    r"""The Multihead Attention Block (MAB) from the `"Set Transformer: A
    Framework for Attention-based Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper.

    .. math::

        \mathrm{MAB}(\mathbf{x}, \mathbf{y}) &= \mathrm{LayerNorm}(\mathbf{h} +
        \mathbf{W} \mathbf{h})

        \mathbf{h} &= \mathrm{LayerNorm}(\mathbf{x} +
        \mathrm{Multihead}(\mathbf{x}, \mathbf{y}, \mathbf{y}))

    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        norm (str, optional): If set to :obj:`False`, will not apply layer
            normalization. (default: :obj:`True`)
        dropout (float, optional): Dropout probability of attention weights.
            (default: :obj:`0`)
        device (torch.device, optional): The device of the module.
            (default: :obj:`None`)
    """
    def __init__(self, channels: int, heads: int = 1, layer_norm: bool = True,
                 dropout: float = 0.0, device: Optional[torch.device] = None):
        super().__init__()

        self.channels = channels
        self.heads = heads
        self.dropout = dropout

        self.attn = MultiheadAttention(
            channels,
            heads,
            batch_first=True,
            dropout=dropout,
            device=device,
        )
        self.lin = Linear(channels, channels, device=device)
        self.layer_norm1 = LayerNorm(channels,
                                     device=device) if layer_norm else None
        self.layer_norm2 = LayerNorm(channels,
                                     device=device) if layer_norm else None

    def reset_parameters(self):
        self.attn._reset_parameters()
        self.lin.reset_parameters()
        if self.layer_norm1 is not None:
            self.layer_norm1.reset_parameters()
        if self.layer_norm2 is not None:
            self.layer_norm2.reset_parameters()

    def forward(self, x: Tensor, y: Tensor, x_mask: Optional[Tensor] = None,
                y_mask: Optional[Tensor] = None) -> Tensor:
        """"""  # noqa: D419
        if y_mask is not None:
            y_mask = ~y_mask

        out, _ = self.attn(x, y, y, y_mask, need_weights=False)

        if x_mask is not None:
            out[~x_mask] = 0.

        out = out + x

        if self.layer_norm1 is not None:
            out = self.layer_norm1(out)

        out = out + self.lin(out).relu()

        if self.layer_norm2 is not None:
            out = self.layer_norm2(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'heads={self.heads}, '
                f'layer_norm={self.layer_norm1 is not None}, '
                f'dropout={self.dropout})')

# ---- InducedSetAttentionBlock (target) ----
class InducedSetAttentionBlock(torch.nn.Module):
    r"""The Induced Set Attention Block (SAB) from the `"Set Transformer: A
    Framework for Attention-based Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper.

    .. math::

        \mathrm{ISAB}(\mathbf{X}) &= \mathrm{MAB}(\mathbf{x}, \mathbf{h})

        \mathbf{h} &= \mathrm{MAB}(\mathbf{I}, \mathbf{x})

    where :math:`\mathbf{I}` denotes :obj:`num_induced_points` learnable
    vectors.

    Args:
        channels (int): Size of each input sample.
        num_induced_points (int): Number of induced points.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        norm (str, optional): If set to :obj:`False`, will not apply layer
            normalization. (default: :obj:`True`)
        dropout (float, optional): Dropout probability of attention weights.
            (default: :obj:`0`)
    """
    def __init__(self, channels: int, num_induced_points: int, heads: int = 1,
                 layer_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.ind = Parameter(torch.empty(1, num_induced_points, channels))
        self.mab1 = MultiheadAttentionBlock(channels, heads, layer_norm,
                                            dropout)
        self.mab2 = MultiheadAttentionBlock(channels, heads, layer_norm,
                                            dropout)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.ind)
        self.mab1.reset_parameters()
        self.mab2.reset_parameters()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        h = self.mab1(self.ind.expand(x.size(0), -1, -1), x, y_mask=mask)
        return self.mab2(x, h, x_mask=mask)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.ind.size(2)}, '
                f'num_induced_points={self.ind.size(1)}, '
                f'heads={self.mab1.heads}, '
                f'layer_norm={self.mab1.layer_norm1 is not None}, '
                f'dropout={self.mab1.dropout})')

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
        self.induced_set_attention = InducedSetAttentionBlock(channels=32, num_induced_points=4, heads=1, layer_norm=True, dropout=0.0)
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).transpose(1, 2)
        x = self.induced_set_attention(x)
        x = x.mean(dim=1)
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
