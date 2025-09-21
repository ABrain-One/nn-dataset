# Auto-generated single-file for DeformableTransformerDecoderLayer
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# H_, W_
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional as _Optional

H_, W_ = 8, 8

# ---- torch.nn.init._no_grad_fill_ ----
def _no_grad_fill_(tensor: Tensor, val: float) -> Tensor:
    with torch.no_grad():
        return tensor.fill_(val)

# ---- torch.nn.init.constant_ ----
def constant_(tensor: Tensor, val: float) -> Tensor:
    r"""Fill the input Tensor with the value :math:`\text{val}`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        val: the value to fill the tensor with

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.constant_(w, 0.3)
    """
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            constant_, (tensor,), tensor=tensor, val=val
        )
    return _no_grad_fill_(tensor, val)

# ---- torch.nn.init._calculate_fan_in_and_fan_out ----
def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

# ---- torch.nn.init._no_grad_uniform_ ----
def _no_grad_uniform_(
    tensor: Tensor, a: float, b: float, generator: _Optional[torch.Generator] = None
) -> Tensor:
    with torch.no_grad():
        return tensor.uniform_(a, b, generator=generator)

# ---- torch.nn.init.xavier_uniform_ ----
def xavier_uniform_(
    tensor: Tensor,
    gain: float = 1.0,
    generator: _Optional[torch.Generator] = None,
) -> Tensor:
    r"""Fill the input `Tensor` with values using a Xavier uniform distribution.

    The method is described in `Understanding the difficulty of training
    deep feedforward neural networks` - Glorot, X. & Bengio, Y. (2010).
    The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain("relu"))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a, generator)

# ---- ultralytics.nn.modules.utils.multi_scale_deformable_attn_pytorch ----
def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Implement multi-scale deformable attention in PyTorch.

    This function performs deformable attention across multiple feature map scales, allowing the model to attend to
    different spatial locations with learned offsets.

    Args:
        value (torch.Tensor): The value tensor with shape (bs, num_keys, num_heads, embed_dims).
        value_spatial_shapes (torch.Tensor): Spatial shapes of the value tensor with shape (num_levels, 2).
        sampling_locations (torch.Tensor): The sampling locations with shape
            (bs, num_queries, num_heads, num_levels, num_points, 2).
        attention_weights (torch.Tensor): The attention weights with shape
            (bs, num_queries, num_heads, num_levels, num_points).

    Returns:
        (torch.Tensor): The output tensor with shape (bs, num_queries, embed_dims).

    References:
        https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()

# ---- ultralytics.nn.modules.transformer.MSDeformAttn ----
class MSDeformAttn(nn.Module):
    """
    Multiscale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    This module implements multiscale deformable attention that can attend to features at multiple scales
    with learnable sampling locations and attention weights.

    Attributes:
        im2col_step (int): Step size for im2col operations.
        d_model (int): Model dimension.
        n_levels (int): Number of feature levels.
        n_heads (int): Number of attention heads.
        n_points (int): Number of sampling points per attention head per feature level.
        sampling_offsets (nn.Linear): Linear layer for generating sampling offsets.
        attention_weights (nn.Linear): Linear layer for generating attention weights.
        value_proj (nn.Linear): Linear layer for projecting values.
        output_proj (nn.Linear): Linear layer for projecting output.

    References:
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        """
        Initialize MSDeformAttn with the given parameters.

        Args:
            d_model (int): Model dimension.
            n_levels (int): Number of feature levels.
            n_heads (int): Number of attention heads.
            n_points (int): Number of sampling points per attention head per feature level.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, "`d_model` must be divisible by `n_heads`"

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        refer_bbox: torch.Tensor,
        value: torch.Tensor,
        value_shapes: list,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform forward pass for multiscale deformable attention.

        Args:
            query (torch.Tensor): Query tensor with shape [bs, query_length, C].
            refer_bbox (torch.Tensor): Reference bounding boxes with shape [bs, query_length, n_levels, 2],
                range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area.
            value (torch.Tensor): Value tensor with shape [bs, value_length, C].
            value_shapes (list): List with shape [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})].
            value_mask (torch.Tensor, optional): Mask tensor with shape [bs, value_length], True for non-padding
                elements, False for padding elements.

        Returns:
            (torch.Tensor): Output tensor with shape [bs, Length_{query}, C].

        References:
            https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {num_points}.")
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)

# ---- DeformableTransformerDecoderLayer (target) ----
class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    This class implements a single decoder layer with self-attention, cross-attention using multiscale deformable
    attention, and a feedforward network.

    Attributes:
        self_attn (nn.MultiheadAttention): Self-attention module.
        dropout1 (nn.Dropout): Dropout after self-attention.
        norm1 (nn.LayerNorm): Layer normalization after self-attention.
        cross_attn (MSDeformAttn): Cross-attention module.
        dropout2 (nn.Dropout): Dropout after cross-attention.
        norm2 (nn.LayerNorm): Layer normalization after cross-attention.
        linear1 (nn.Linear): First linear layer in the feedforward network.
        act (nn.Module): Activation function.
        dropout3 (nn.Dropout): Dropout in the feedforward network.
        linear2 (nn.Linear): Second linear layer in the feedforward network.
        dropout4 (nn.Dropout): Dropout after the feedforward network.
        norm3 (nn.LayerNorm): Layer normalization after the feedforward network.

    References:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        n_levels: int = 4,
        n_points: int = 4,
    ):
        """
        Initialize the DeformableTransformerDecoderLayer with the given parameters.

        Args:
            d_model (int): Model dimension.
            n_heads (int): Number of attention heads.
            d_ffn (int): Dimension of the feedforward network.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            n_levels (int): Number of feature levels.
            n_points (int): Number of sampling points.
        """
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the Feed-Forward Network part of the layer.

        Args:
            tgt (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after FFN.
        """
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform the forward pass through the entire decoder layer.

        Args:
            embed (torch.Tensor): Input embeddings.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            feats (torch.Tensor): Feature maps.
            shapes (list): Feature shapes.
            padding_mask (torch.Tensor, optional): Padding mask.
            attn_mask (torch.Tensor, optional): Attention mask.
            query_pos (torch.Tensor, optional): Query position embeddings.

        Returns:
            (torch.Tensor): Output tensor after decoder layer.
        """
        # Self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attn_mask=attn_mask)[
            0
        ].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # Cross attention
        tgt = self.cross_attn(
            self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask
        )
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self.forward_ffn(embed)

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
        
        self.decoder_layer = DeformableTransformerDecoderLayer(
            d_model=32,
            n_heads=4,
            d_ffn=128,
            dropout=0.1,
            n_levels=1,
            n_points=4
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
