# Auto-generated single-file for PolynormerAttention
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# ---- original imports from contributing modules ----

# ---- PolynormerAttention (target) ----
class PolynormerAttention(torch.nn.Module):
    r"""The polynomial-expressive attention mechanism from the
    `"Polynormer: Polynomial-Expressive Graph Transformer in Linear Time"
    <https://arxiv.org/abs/2403.01232>`_ paper.

    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of parallel attention heads.
        head_channels (int, optional): Size of each attention head.
            (default: :obj:`64.`)
        beta (float, optional): Polynormer beta initialization.
            (default: :obj:`0.9`)
        qkv_bias (bool, optional): If specified, add bias to query, key
            and value in the self attention. (default: :obj:`False`)
        qk_shared (bool optional): Whether weight of query and key are shared.
            (default: :obj:`True`)
        dropout (float, optional): Dropout probability of the final
            attention output. (default: :obj:`0.0`)
    """
    def __init__(
        self,
        channels: int,
        heads: int,
        head_channels: int = 64,
        beta: float = 0.9,
        qkv_bias: bool = False,
        qk_shared: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.head_channels = head_channels
        self.heads = heads
        self.beta = beta
        self.qk_shared = qk_shared

        inner_channels = heads * head_channels
        self.h_lins = torch.nn.Linear(channels, inner_channels)
        if not self.qk_shared:
            self.q = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.k = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.v = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.lns = torch.nn.LayerNorm(inner_channels)
        self.lin_out = torch.nn.Linear(inner_channels, inner_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """
        B, N, *_ = x.shape
        h = self.h_lins(x)
        k = self.k(x).sigmoid().view(B, N, self.head_channels, self.heads)
        if self.qk_shared:
            q = k
        else:
            q = F.sigmoid(self.q(x)).view(B, N, self.head_channels, self.heads)
        v = self.v(x).view(B, N, self.head_channels, self.heads)

        if mask is not None:
            mask = mask[:, :, None, None]
            v.masked_fill_(~mask, 0.)

        # numerator
        kv = torch.einsum('bndh, bnmh -> bdmh', k, v)
        num = torch.einsum('bndh, bdmh -> bnmh', q, kv)

        # denominator
        k_sum = torch.einsum('bndh -> bdh', k)
        den = torch.einsum('bndh, bdh -> bnh', q, k_sum).unsqueeze(2)

        # linear global attention based on kernel trick
        x = (num / (den + 1e-6)).reshape(B, N, -1)
        x = self.lns(x) * (h + self.beta)
        x = F.relu(self.lin_out(x))
        x = self.dropout(x)

        return x

    def reset_parameters(self) -> None:
        self.h_lins.reset_parameters()
        if not self.qk_shared:
            self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()
        self.lns.reset_parameters()
        self.lin_out.reset_parameters()

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'heads={self.heads}, '
                f'head_channels={self.head_channels})')

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
        self.polynormer_attention = PolynormerAttention(
            channels=32,
            heads=2,
            head_channels=16,
            beta=0.9,
            qkv_bias=False,
            qk_shared=True,
            dropout=0.0
        )
        self.classifier = torch.nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=False),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=False),
        ]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.polynormer_attention(x)
        x = x.mean(dim=1)
        return self.classifier(x)

    def train_setup(self, prm):
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
