# Auto-generated single-file for PerformerAttention
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
from torch import Tensor
from typing import Optional, Callable
import math
from typing import Optional
from typing import Callable

# ---- torch_geometric.nn.attention.performer.generalized_kernel ----
def generalized_kernel(
        x: Tensor,
        mat: Tensor,
        kernel: Callable = torch.nn.ReLU(),
        epsilon: float = 0.001,
) -> Tensor:
    batch_size, num_heads = x.size()[:2]
    projection = mat.t().expand(batch_size, num_heads, -1, -1)
    x = x @ projection
    out = kernel(x) + epsilon
    return out

# ---- torch_geometric.nn.attention.performer.linear_attention ----
def linear_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    r"""Efficient attention mechanism from the
    `"Rethinking Attention with Performers"
    <https://arxiv.org/abs/2009.14794>`_ paper.

    .. math::
        \mathbf{\hat{D}}^{-1}(\mathbf{Q}'((\mathbf{K}')^{\top} \mathbf{V}))

    """
    D_inv = 1.0 / (q @ k.sum(dim=-2).unsqueeze(-1))
    kv = k.transpose(-2, -1) @ v
    qkv = q @ kv
    out = torch.einsum('...L,...Ld->...Ld', D_inv.squeeze(-1), qkv)
    return out

# ---- torch_geometric.nn.attention.performer._orthogonal_matrix ----
def _orthogonal_matrix(dim: int) -> Tensor:
    r"""Get an orthogonal matrix by applying QR decomposition."""
    # Random matrix from normal distribution
    mat = torch.randn((dim, dim))
    # QR decomposition to two orthogonal matrices
    q, _ = torch.linalg.qr(mat.cpu(), mode='reduced')
    return q.t()

# ---- torch_geometric.nn.attention.performer.orthogonal_matrix ----
def orthogonal_matrix(num_rows: int, num_cols: int) -> Tensor:
    r"""Generate an orthogonal matrix with `num_rows` rows
    and `num_cols` columns.
    """
    num_full_blocks = int(num_rows / num_cols)
    blocks = []
    for _ in range(num_full_blocks):
        q = _orthogonal_matrix(num_cols)
        blocks.append(q)
    remain_rows = num_rows - num_full_blocks * num_cols
    if remain_rows > 0:
        q = _orthogonal_matrix(num_cols)
        blocks.append(q[:remain_rows])
    mat = torch.cat(blocks)
    # multiplier = torch.randn((num_rows, num_cols)).norm(dim=1)
    # scaler = torch.diag(multiplier)
    # mat = scaler @ mat
    return mat

# ---- torch_geometric.nn.attention.performer.PerformerProjection ----
class PerformerProjection(torch.nn.Module):
    r"""The fast attention that uses a projection matrix
    from the `"Rethinking Attention with Performers"
    <https://arxiv.org/abs/2009.14794>`_ paper. This class
    projects :math:`\mathbf{Q}` and :math:`\mathbf{K}` matrices
    with specified kernel.

    Args:
        num_cols (int): Projection matrix number of columns.
        kernel (Callable, optional): Kernels for generalized attention.
            If not specified, `ReLU` kernel will be used.
            (default: :obj:`torch.nn.ReLU()`)
    """
    def __init__(self, num_cols: int, kernel: Callable = torch.nn.ReLU()):
        super().__init__()
        num_rows = int(num_cols * math.log(num_cols))
        self.num_rows = num_rows
        self.num_cols = num_cols
        # Generate an orthogonal projection matrix
        # with the shape (num_rows, num_cols)
        projection_matrix = orthogonal_matrix(self.num_rows, self.num_cols)
        self.register_buffer('projection_matrix', projection_matrix)
        assert kernel is not None
        self.kernel = kernel

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = generalized_kernel(q, self.projection_matrix, self.kernel)
        k = generalized_kernel(k, self.projection_matrix, self.kernel)
        out = linear_attention(q, k, v)
        return out

# ---- PerformerAttention (target) ----
class PerformerAttention(torch.nn.Module):
    r"""The linear scaled attention mechanism from the
    `"Rethinking Attention with Performers"
    <https://arxiv.org/abs/2009.14794>`_ paper.

    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of parallel attention heads.
        head_channels (int, optional): Size of each attention head.
            (default: :obj:`64.`)
        kernel (Callable, optional): Kernels for generalized attention.
            If not specified, `ReLU` kernel will be used.
            (default: :obj:`torch.nn.ReLU()`)
        qkv_bias (bool, optional): If specified, add bias to query, key
            and value in the self attention. (default: :obj:`False`)
        attn_out_bias (bool, optional): If specified, add bias to the
            attention output. (default: :obj:`True`)
        dropout (float, optional): Dropout probability of the final
            attention output. (default: :obj:`0.0`)

    """
    def __init__(
        self,
        channels: int,
        heads: int,
        head_channels: int = 64,
        kernel: Callable = torch.nn.ReLU(),
        qkv_bias: bool = False,
        attn_out_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert channels % heads == 0
        if head_channels is None:
            head_channels = channels // heads

        self.heads = heads
        self.head_channels = head_channels
        self.kernel = kernel
        self.fast_attn = PerformerProjection(head_channels, kernel)

        inner_channels = head_channels * heads
        self.q = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.k = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.v = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.attn_out = torch.nn.Linear(inner_channels, channels,
                                        bias=attn_out_bias)
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
        q, k, v = self.q(x), self.k(x), self.v(x)
        # Reshape and permute q, k and v to proper shape
        # (B, N, num_heads * head_channels) to (b, num_heads, n, head_channels)
        q, k, v = map(
            lambda t: t.reshape(B, N, self.heads, self.head_channels).permute(
                0, 2, 1, 3), (q, k, v))
        if mask is not None:
            mask = mask[:, None, :, None]
            v.masked_fill_(~mask, 0.)
        out = self.fast_attn(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        out = self.attn_out(out)
        out = self.dropout(out)
        return out

    def redraw_projection_matrix(self):
        r"""As described in the paper, periodically redraw
        examples to improve overall approximation of attention.
        """
        num_rows = self.fast_attn.num_rows
        num_cols = self.fast_attn.num_cols
        projection_matrix = orthogonal_matrix(num_rows, num_cols)
        self.fast_attn.projection_matrix.copy_(projection_matrix)
        del projection_matrix

    def _reset_parameters(self):
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()
        self.attn_out.reset_parameters()
        self.redraw_projection_matrix()

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'heads={self.heads}, '
                f'head_channels={self.head_channels} '
                f'kernel={self.kernel})')

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(torch.nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, 8, 3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(4, 4),
            torch.nn.Conv2d(8, 16, 3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(4, 4),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.performer_attention = PerformerAttention(
            channels=32,
            heads=1,
            head_channels=8,
            kernel=torch.nn.ReLU(),
            qkv_bias=False,
            attn_out_bias=True,
            dropout=0.1
        )
        
        self.classifier = torch.nn.Linear(32, self.num_classes)
        
    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).transpose(1, 2)
        x = self.performer_attention(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)
    
    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)
