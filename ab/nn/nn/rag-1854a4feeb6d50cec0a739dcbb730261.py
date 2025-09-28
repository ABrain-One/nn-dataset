# Auto-generated single-file for SGFormerAttention
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# ---- original imports from contributing modules ----

# ---- SGFormerAttention (target) ----
class SGFormerAttention(torch.nn.Module):
    r"""The simple global attention mechanism from the
    `"SGFormer: Simplifying and Empowering Transformers for
    Large-Graph Representations"
    <https://arxiv.org/abs/2306.10759>`_ paper.

    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of parallel attention heads.
            (default: :obj:`1.`)
        head_channels (int, optional): Size of each attention head.
            (default: :obj:`64.`)
        qkv_bias (bool, optional): If specified, add bias to query, key
            and value in the self attention. (default: :obj:`False`)
    """
    def __init__(
        self,
        channels: int,
        heads: int = 1,
        head_channels: int = 64,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert channels % heads == 0
        if head_channels is None:
            head_channels = channels // heads

        self.heads = heads
        self.head_channels = head_channels

        inner_channels = head_channels * heads
        self.q = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.k = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.v = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)

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
        qs, ks, vs = self.q(x), self.k(x), self.v(x)
        # reshape and permute q, k and v to proper shape
        # (b, n, num_heads * head_channels) to (b, n, num_heads, head_channels)
        qs, ks, vs = map(
            lambda t: t.reshape(B, N, self.heads, self.head_channels),
            (qs, ks, vs))

        if mask is not None:
            mask = mask[:, :, None, None]
            vs.masked_fill_(~mask, 0.)
        # replace 0's with epsilon
        epsilon = 1e-6
        qs[qs == 0] = epsilon
        ks[ks == 0] = epsilon
        # normalize input, shape not changed
        qs, ks = map(
            lambda t: t / torch.linalg.norm(t, ord=2, dim=-1, keepdim=True),
            (qs, ks))

        # numerator
        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
        attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)
        attention_num += N * vs

        # denominator
        all_ones = torch.ones([B, N]).to(ks.device)
        ks_sum = torch.einsum("blhm,bl->bhm", ks, all_ones)
        attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)
        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(attention_normalizer,
                                               len(attention_normalizer.shape))
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer

        return attn_output.mean(dim=2)

    def reset_parameters(self):
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'heads={self.heads}, '
                f'head_channels={self.head_channels})')

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
        self.sg_attention = SGFormerAttention(channels=64, heads=4, head_channels=16, qkv_bias=True)
        self.classifier = nn.Linear(16, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)
        x = self.sg_attention(x)
        x = x.mean(dim=1)
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
