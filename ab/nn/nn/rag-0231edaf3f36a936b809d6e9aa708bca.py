# Auto-generated single-file for MultiscaleBlock
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Optional, Callable
import math
from collections.abc import Sequence
from types import FunctionType
from typing import Optional
from typing import TypeVar
T = TypeVar("T")
from typing import Any
from typing import Callable
from dataclasses import dataclass

# ---- torchvision.models.video.mvit.MSBlockConfig ----
class MSBlockConfig:
    num_heads: int
    input_channels: int
    output_channels: int
    kernel_q: list[int]
    kernel_kv: list[int]
    stride_q: list[int]
    stride_kv: list[int]

# ---- torchvision.utils._log_api_usage_once ----
def _log_api_usage_once(obj: Any) -> None:
    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;

    Args:
        obj (class instance or method): an object to extract info from.
    """
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")

# ---- torchvision.ops.misc.MLP ----
class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
        _log_api_usage_once(self)

# ---- torchvision.models.video.mvit._squeeze ----
def _squeeze(x: torch.Tensor, target_dim: int, expand_dim: int, tensor_dim: int) -> torch.Tensor:
    if tensor_dim == target_dim - 1:
        x = x.squeeze(expand_dim)
    return x

# ---- torchvision.models.video.mvit._unsqueeze ----
def _unsqueeze(x: torch.Tensor, target_dim: int, expand_dim: int) -> tuple[torch.Tensor, int]:
    tensor_dim = x.dim()
    if tensor_dim == target_dim - 1:
        x = x.unsqueeze(expand_dim)
    elif tensor_dim != target_dim:
        raise ValueError(f"Unsupported input dimension {x.shape}")
    return x, tensor_dim

# ---- torchvision.models.video.mvit.Pool ----
class Pool(nn.Module):
    def __init__(
        self,
        pool: nn.Module,
        norm: Optional[nn.Module],
        activation: Optional[nn.Module] = None,
        norm_before_pool: bool = False,
    ) -> None:
        super().__init__()
        self.pool = pool
        layers = []
        if norm is not None:
            layers.append(norm)
        if activation is not None:
            layers.append(activation)
        self.norm_act = nn.Sequential(*layers) if layers else None
        self.norm_before_pool = norm_before_pool

    def forward(self, x: torch.Tensor, thw: tuple[int, int, int]) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x, tensor_dim = _unsqueeze(x, 4, 1)

        # Separate the class token and reshape the input
        class_token, x = torch.tensor_split(x, indices=(1,), dim=2)
        x = x.transpose(2, 3)
        B, N, C = x.shape[:3]
        x = x.reshape((B * N, C) + thw).contiguous()

        # normalizing prior pooling is useful when we use BN which can be absorbed to speed up inference
        if self.norm_before_pool and self.norm_act is not None:
            x = self.norm_act(x)

        # apply the pool on the input and add back the token
        x = self.pool(x)
        T, H, W = x.shape[2:]
        x = x.reshape(B, N, C, -1).transpose(2, 3)
        x = torch.cat((class_token, x), dim=2)

        if not self.norm_before_pool and self.norm_act is not None:
            x = self.norm_act(x)

        x = _squeeze(x, 4, 1, tensor_dim)
        return x, (T, H, W)

# ---- torchvision.models.video.mvit._interpolate ----
def _interpolate(embedding: torch.Tensor, d: int) -> torch.Tensor:
    if embedding.shape[0] == d:
        return embedding

    return (
        nn.functional.interpolate(
            embedding.permute(1, 0).unsqueeze(0),
            size=d,
            mode="linear",
        )
        .squeeze(0)
        .permute(1, 0)
    )

# ---- torchvision.models.video.mvit._add_rel_pos ----
def _add_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    q_thw: tuple[int, int, int],
    k_thw: tuple[int, int, int],
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    rel_pos_t: torch.Tensor,
) -> torch.Tensor:
    # Modified code from: https://github.com/facebookresearch/SlowFast/commit/1aebd71a2efad823d52b827a3deaf15a56cf4932
    q_t, q_h, q_w = q_thw
    k_t, k_h, k_w = k_thw
    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1)
    dt = int(2 * max(q_t, k_t) - 1)

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = torch.arange(q_h)[:, None] * q_h_ratio - (torch.arange(k_h)[None, :] + (1.0 - k_h)) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = torch.arange(q_w)[:, None] * q_w_ratio - (torch.arange(k_w)[None, :] + (1.0 - k_w)) * k_w_ratio
    q_t_ratio = max(k_t / q_t, 1.0)
    k_t_ratio = max(q_t / k_t, 1.0)
    dist_t = torch.arange(q_t)[:, None] * q_t_ratio - (torch.arange(k_t)[None, :] + (1.0 - k_t)) * k_t_ratio

    # Interpolate rel pos if needed.
    rel_pos_h = _interpolate(rel_pos_h, dh)
    rel_pos_w = _interpolate(rel_pos_w, dw)
    rel_pos_t = _interpolate(rel_pos_t, dt)
    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]
    Rt = rel_pos_t[dist_t.long()]

    B, n_head, _, dim = q.shape

    r_q = q[:, :, 1:].reshape(B, n_head, q_t, q_h, q_w, dim)
    rel_h_q = torch.einsum("bythwc,hkc->bythwk", r_q, Rh)  # [B, H, q_t, qh, qw, k_h]
    rel_w_q = torch.einsum("bythwc,wkc->bythwk", r_q, Rw)  # [B, H, q_t, qh, qw, k_w]
    # [B, H, q_t, q_h, q_w, dim] -> [q_t, B, H, q_h, q_w, dim] -> [q_t, B*H*q_h*q_w, dim]
    r_q = r_q.permute(2, 0, 1, 3, 4, 5).reshape(q_t, B * n_head * q_h * q_w, dim)
    # [q_t, B*H*q_h*q_w, dim] * [q_t, dim, k_t] = [q_t, B*H*q_h*q_w, k_t] -> [B*H*q_h*q_w, q_t, k_t]
    rel_q_t = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(0, 1)
    # [B*H*q_h*q_w, q_t, k_t] -> [B, H, q_t, q_h, q_w, k_t]
    rel_q_t = rel_q_t.view(B, n_head, q_h, q_w, q_t, k_t).permute(0, 1, 4, 2, 3, 5)

    # Combine rel pos.
    rel_pos = (
        rel_h_q[:, :, :, :, :, None, :, None]
        + rel_w_q[:, :, :, :, :, None, None, :]
        + rel_q_t[:, :, :, :, :, :, None, None]
    ).reshape(B, n_head, q_t * q_h * q_w, k_t * k_h * k_w)

    # Add it to attention
    attn[:, :, 1:, 1:] += rel_pos

    return attn

# ---- torchvision.models.video.mvit._add_shortcut ----
def _add_shortcut(x: torch.Tensor, shortcut: torch.Tensor, residual_with_cls_embed: bool):
    if residual_with_cls_embed:
        x.add_(shortcut)
    else:
        x[:, :, 1:, :] += shortcut[:, :, 1:, :]
    return x

# ---- torchvision.models.video.mvit._prod ----
def _prod(s: Sequence[int]) -> int:
    product = 1
    for v in s:
        product *= v
    return product

# ---- torchvision.models.video.mvit.MultiscaleAttention ----
class MultiscaleAttention(nn.Module):
    def __init__(
        self,
        input_size: list[int],
        embed_dim: int,
        output_dim: int,
        num_heads: int,
        kernel_q: list[int],
        kernel_kv: list[int],
        stride_q: list[int],
        stride_kv: list[int],
        residual_pool: bool,
        residual_with_cls_embed: bool,
        rel_pos_embed: bool,
        dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scaler = 1.0 / math.sqrt(self.head_dim)
        self.residual_pool = residual_pool
        self.residual_with_cls_embed = residual_with_cls_embed

        self.qkv = nn.Linear(embed_dim, 3 * output_dim)
        layers: list[nn.Module] = [nn.Linear(output_dim, output_dim)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout, inplace=True))
        self.project = nn.Sequential(*layers)

        self.pool_q: Optional[nn.Module] = None
        if _prod(kernel_q) > 1 or _prod(stride_q) > 1:
            padding_q = [int(q // 2) for q in kernel_q]
            self.pool_q = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_q,  # type: ignore[arg-type]
                    stride=stride_q,  # type: ignore[arg-type]
                    padding=padding_q,  # type: ignore[arg-type]
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim),
            )

        self.pool_k: Optional[nn.Module] = None
        self.pool_v: Optional[nn.Module] = None
        if _prod(kernel_kv) > 1 or _prod(stride_kv) > 1:
            padding_kv = [int(kv // 2) for kv in kernel_kv]
            self.pool_k = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_kv,  # type: ignore[arg-type]
                    stride=stride_kv,  # type: ignore[arg-type]
                    padding=padding_kv,  # type: ignore[arg-type]
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim),
            )
            self.pool_v = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_kv,  # type: ignore[arg-type]
                    stride=stride_kv,  # type: ignore[arg-type]
                    padding=padding_kv,  # type: ignore[arg-type]
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim),
            )

        self.rel_pos_h: Optional[nn.Parameter] = None
        self.rel_pos_w: Optional[nn.Parameter] = None
        self.rel_pos_t: Optional[nn.Parameter] = None
        if rel_pos_embed:
            size = max(input_size[1:])
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            spatial_dim = 2 * max(q_size, kv_size) - 1
            temporal_dim = 2 * input_size[0] - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(spatial_dim, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(spatial_dim, self.head_dim))
            self.rel_pos_t = nn.Parameter(torch.zeros(temporal_dim, self.head_dim))
            nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
            nn.init.trunc_normal_(self.rel_pos_w, std=0.02)
            nn.init.trunc_normal_(self.rel_pos_t, std=0.02)

    def forward(self, x: torch.Tensor, thw: tuple[int, int, int]) -> tuple[torch.Tensor, tuple[int, int, int]]:
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(1, 3).unbind(dim=2)

        if self.pool_k is not None:
            k, k_thw = self.pool_k(k, thw)
        else:
            k_thw = thw
        if self.pool_v is not None:
            v = self.pool_v(v, thw)[0]
        if self.pool_q is not None:
            q, thw = self.pool_q(q, thw)

        attn = torch.matmul(self.scaler * q, k.transpose(2, 3))
        if self.rel_pos_h is not None and self.rel_pos_w is not None and self.rel_pos_t is not None:
            attn = _add_rel_pos(
                attn,
                q,
                thw,
                k_thw,
                self.rel_pos_h,
                self.rel_pos_w,
                self.rel_pos_t,
            )
        attn = attn.softmax(dim=-1)

        x = torch.matmul(attn, v)
        if self.residual_pool:
            _add_shortcut(x, q, self.residual_with_cls_embed)
        x = x.transpose(1, 2).reshape(B, -1, self.output_dim)
        x = self.project(x)

        return x, thw

# ---- torchvision.ops.stochastic_depth.stochastic_depth ----
def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(stochastic_depth)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise

# ---- torchvision.ops.stochastic_depth.StochasticDepth ----
class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s

# ---- MultiscaleBlock (target) ----
class MultiscaleBlock(nn.Module):
    def __init__(
        self,
        input_size: list[int],
        cnf: MSBlockConfig,
        residual_pool: bool,
        residual_with_cls_embed: bool,
        rel_pos_embed: bool,
        proj_after_attn: bool,
        dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.proj_after_attn = proj_after_attn

        self.pool_skip: Optional[nn.Module] = None
        if _prod(cnf.stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in cnf.stride_q]
            padding_skip = [int(k // 2) for k in kernel_skip]
            self.pool_skip = Pool(
                nn.MaxPool3d(kernel_skip, stride=cnf.stride_q, padding=padding_skip), None  # type: ignore[arg-type]
            )

        attn_dim = cnf.output_channels if proj_after_attn else cnf.input_channels

        self.norm1 = norm_layer(cnf.input_channels)
        self.norm2 = norm_layer(attn_dim)
        self.needs_transposal = isinstance(self.norm1, nn.BatchNorm1d)

        self.attn = MultiscaleAttention(
            input_size,
            cnf.input_channels,
            attn_dim,
            cnf.num_heads,
            kernel_q=cnf.kernel_q,
            kernel_kv=cnf.kernel_kv,
            stride_q=cnf.stride_q,
            stride_kv=cnf.stride_kv,
            rel_pos_embed=rel_pos_embed,
            residual_pool=residual_pool,
            residual_with_cls_embed=residual_with_cls_embed,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.mlp = MLP(
            attn_dim,
            [4 * attn_dim, cnf.output_channels],
            activation_layer=nn.GELU,
            dropout=dropout,
            inplace=None,
        )

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

        self.project: Optional[nn.Module] = None
        if cnf.input_channels != cnf.output_channels:
            self.project = nn.Linear(cnf.input_channels, cnf.output_channels)

    def forward(self, x: torch.Tensor, thw: tuple[int, int, int]) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x_norm1 = self.norm1(x.transpose(1, 2)).transpose(1, 2) if self.needs_transposal else self.norm1(x)
        x_attn, thw_new = self.attn(x_norm1, thw)
        x = x if self.project is None or not self.proj_after_attn else self.project(x_norm1)
        x_skip = x if self.pool_skip is None else self.pool_skip(x, thw)[0]
        x = x_skip + self.stochastic_depth(x_attn)

        x_norm2 = self.norm2(x.transpose(1, 2)).transpose(1, 2) if self.needs_transposal else self.norm2(x)
        x_proj = x if self.project is None or self.proj_after_attn else self.project(x_norm2)

        return x_proj + self.stochastic_depth(self.mlp(x_norm2)), thw_new

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
        
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(8, 8),
        )
        
        cnf = MSBlockConfig()
        cnf.num_heads = 1
        cnf.input_channels = 8
        cnf.output_channels = 8
        cnf.kernel_q = [1, 1, 1]
        cnf.kernel_kv = [1, 1, 1]
        cnf.stride_q = [1, 1, 1]
        cnf.stride_kv = [1, 1, 1]
        
        self.multiscale_block = MultiscaleBlock(
            input_size=[1, 4, 4],
            cnf=cnf,
            residual_pool=True,
            residual_with_cls_embed=False,
            rel_pos_embed=False,
            proj_after_attn=True,
            dropout=0.0,
            stochastic_depth_prob=0.0,
            norm_layer=nn.LayerNorm
        )
        self.classifier = nn.Linear(8, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x, _ = self.multiscale_block(x, (1, H, W))
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
            output = self.forward(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)
