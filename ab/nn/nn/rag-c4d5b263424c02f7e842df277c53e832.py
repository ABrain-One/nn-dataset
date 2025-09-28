# Auto-generated single-file for VLFuse
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# checkpoint
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from typing import Optional
from torch.utils.checkpoint import checkpoint
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass
from typing import Tuple

# ---- mmcv.cnn.bricks.drop.drop_path ----
def drop_path(x: torch.Tensor,
              drop_prob: float = 0.,
              training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py
    # noqa: E501
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output

# ---- mmcv.cnn.bricks.drop.DropPath ----
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

# ---- mmdet.models.utils.vlfuse_helper.permute_and_flatten ----
def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int,
                        W: int) -> Tensor:
    """Permute and then flatten a tensor,

       from size (N, A, C, H, W) to (N, H * W * A, C).

    Args:
        layer (Tensor): Tensor of shape (N, C, H, W).
        N (int): Batch size.
        A (int): Number of attention heads.
        C (int): Number of channels.
        H (int): Height of feature map.
        W (int): Width of feature map.

    Returns:
        Tensor: A Tensor of shape (N, H * W * A, C).
    """
    layer = layer.view(N, A, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer

# ---- mmdet.models.utils.vlfuse_helper.MAX_CLAMP_VALUE ----
MAX_CLAMP_VALUE = 50000

# ---- mmdet.models.utils.vlfuse_helper.BiMultiHeadAttention ----
class BiMultiHeadAttention(nn.Module):
    """Bidirectional fusion Multi-Head Attention layer.

    Args:
        v_dim (int): The dimension of the vision input.
        l_dim (int): The dimension of the language input.
        embed_dim (int): The embedding dimension for the attention operation.
        num_heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
    """

    def __init__(self,
                 v_dim: int,
                 l_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), 'embed_dim must be divisible by num_heads ' \
           f'(got `embed_dim`: {self.embed_dim} ' \
           f'and `num_heads`: {self.num_heads}).'
        self.scale = self.head_dim**(-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = False
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(
        self,
        vision: Tensor,
        lang: Tensor,
        attention_mask_v: Optional[Tensor] = None,
        attention_mask_l: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        bsz, tgt_len, _ = vision.size()

        query_states = self.v_proj(vision) * self.scale
        key_states = self._shape(self.l_proj(lang), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(vision), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(lang), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len,
                                   bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f'Attention weights should be of '
                f'size {(bsz * self.num_heads, tgt_len, src_len)}, '
                f'but is {attn_weights.size()}')

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            # Do not increase -50000, data type half has quite limited range
            attn_weights = torch.clamp(attn_weights, min=-MAX_CLAMP_VALUE)
        if self.clamp_max_for_overflow:
            # Do not increase 50000, data type half has quite limited range
            attn_weights = torch.clamp(attn_weights, max=MAX_CLAMP_VALUE)

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (
            attn_weights_T -
            torch.max(attn_weights_T, dim=-1, keepdim=True)[0])
        if self.clamp_min_for_underflow:
            # Do not increase -50000, data type half has quite limited range
            attn_weights_l = torch.clamp(attn_weights_l, min=-MAX_CLAMP_VALUE)
        if self.clamp_max_for_overflow:
            # Do not increase 50000, data type half has quite limited range
            attn_weights_l = torch.clamp(attn_weights_l, max=MAX_CLAMP_VALUE)

        if attention_mask_v is not None:
            attention_mask_v = (
                attention_mask_v[:, None,
                                 None, :].repeat(1, self.num_heads, 1,
                                                 1).flatten(0, 1))
            attn_weights_l.masked_fill_(attention_mask_v, float('-inf'))

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        if attention_mask_l is not None:
            assert (attention_mask_l.dim() == 2)
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(
                attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError('Attention mask should be of '
                                 f'size {(bsz, 1, tgt_len, src_len)}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len,
                                             src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len,
                                             src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs_v = F.dropout(
            attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(
            attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len,
                                    self.head_dim):
            raise ValueError(
                '`attn_output_v` should be of '
                f'size {(bsz, self.num_heads, tgt_len, self.head_dim)}, '
                f'but is {attn_output_v.size()}')

        if attn_output_l.size() != (bsz * self.num_heads, src_len,
                                    self.head_dim):
            raise ValueError(
                '`attn_output_l` should be of size '
                f'{(bsz, self.num_heads, src_len, self.head_dim)}, '
                f'but is {attn_output_l.size()}')

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len,
                                           self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len,
                                           self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l

# ---- mmdet.models.utils.vlfuse_helper.BiAttentionBlock ----
class BiAttentionBlock(nn.Module):
    """BiAttentionBlock Module:

    First, multi-level visual features are concat; Then the concat visual
    feature and lang feature are fused by attention; Finally the newly visual
    feature are split into multi levels.

    Args:
        v_dim (int): The dimension of the visual features.
        l_dim (int): The dimension of the language feature.
        embed_dim (int): The embedding dimension for the attention operation.
        num_heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        drop_path (float, optional): The drop path probability.
            Defaults to 0.0.
        init_values (float, optional):
            The initial value for the scaling parameter.
            Defaults to 1e-4.
    """

    def __init__(self,
                 v_dim: int,
                 l_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 drop_path: float = .0,
                 init_values: float = 1e-4):
        super().__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim,
            l_dim=l_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout)

        # add layer scale for training stability
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(
            init_values * torch.ones(v_dim), requires_grad=True)
        self.gamma_l = nn.Parameter(
            init_values * torch.ones(l_dim), requires_grad=True)

    def forward(self,
                vf0: Tensor,
                vf1: Tensor,
                vf2: Tensor,
                vf3: Tensor,
                vf4: Tensor,
                lang_feature: Tensor,
                attention_mask_l=None):
        visual_features = [vf0, vf1, vf2, vf3, vf4]
        size_per_level, visual_features_flatten = [], []
        for i, feat_per_level in enumerate(visual_features):
            bs, c, h, w = feat_per_level.shape
            size_per_level.append([h, w])
            feat = permute_and_flatten(feat_per_level, bs, -1, c, h, w)
            visual_features_flatten.append(feat)
        visual_features_flatten = torch.cat(visual_features_flatten, dim=1)
        new_v, new_lang_feature = self.single_attention_call(
            visual_features_flatten,
            lang_feature,
            attention_mask_l=attention_mask_l)
        # [bs, N, C] -> [bs, C, N]
        new_v = new_v.transpose(1, 2).contiguous()

        start = 0
        # fvfs is mean fusion_visual_features
        fvfs = []
        for (h, w) in size_per_level:
            new_v_per_level = new_v[:, :,
                                    start:start + h * w].view(bs, -1, h,
                                                              w).contiguous()
            fvfs.append(new_v_per_level)
            start += h * w

        return fvfs[0], fvfs[1], fvfs[2], fvfs[3], fvfs[4], new_lang_feature

    def single_attention_call(
        self,
        visual: Tensor,
        lang: Tensor,
        attention_mask_v: Optional[Tensor] = None,
        attention_mask_l: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Perform a single attention call between the visual and language
        inputs.

        Args:
        visual (Tensor): The visual input tensor.
        lang (Tensor): The language input tensor.
        attention_mask_v (Optional[Tensor]):
            An optional attention mask tensor for the visual input.
        attention_mask_l (Optional[Tensor]):
            An optional attention mask tensor for the language input.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the updated
                visual and language tensors after the attention call.
        """
        visual = self.layer_norm_v(visual)
        lang = self.layer_norm_l(lang)
        delta_v, delta_l = self.attn(
            visual,
            lang,
            attention_mask_v=attention_mask_v,
            attention_mask_l=attention_mask_l)
        # visual, lang = visual + delta_v, l + delta_l
        visual = visual + self.drop_path(self.gamma_v * delta_v)
        lang = lang + self.drop_path(self.gamma_l * delta_l)
        return visual, lang

# ---- VLFuse (target) ----
class VLFuse(nn.Module):
    """Early Fusion Module.

    Args:
        v_dim (int): Dimension of visual features.
        l_dim (int): Dimension of language features.
        embed_dim (int): The embedding dimension for the attention operation.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        drop_path (float): Drop path probability.
        use_checkpoint (bool): Whether to use PyTorch's checkpoint function.
    """

    def __init__(self,
                 v_dim: int = 256,
                 l_dim: int = 768,
                 embed_dim: int = 2048,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 drop_path: float = 0.0,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.b_attn = BiAttentionBlock(
            v_dim=v_dim,
            l_dim=l_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            drop_path=drop_path,
            init_values=1.0 / 6.0)

    def forward(self, x: dict) -> dict:
        """Forward pass of the VLFuse module."""
        visual_features = x['visual']
        language_dict_features = x['lang']

        if self.use_checkpoint:
            # vf is mean visual_features
            # checkpoint does not allow complex data structures as input,
            # such as list, so we must split them.
            vf0, vf1, vf2, vf3, vf4, language_features = checkpoint.checkpoint(
                self.b_attn, *visual_features,
                language_dict_features['hidden'],
                language_dict_features['masks'])
        else:
            vf0, vf1, vf2, vf3, vf4, language_features = self.b_attn(
                *visual_features, language_dict_features['hidden'],
                language_dict_features['masks'])

        language_dict_features['hidden'] = language_features
        fused_language_dict_features = language_dict_features

        features_dict = {
            'visual': [vf0, vf1, vf2, vf3, vf4],
            'lang': fused_language_dict_features
        }

        return features_dict

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
        
        self.vl_fuse = VLFuse(
            v_dim=64,
            l_dim=64,
            embed_dim=128,
            num_heads=4,
            dropout=0.1,
            drop_path=0.0,
            use_checkpoint=False
        )
        self.classifier = nn.Linear(64 * 4 * 4, self.num_classes)

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
        
        x = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        B, C, H, W = x.shape
        
        visual_features = [x, x, x, x, x]
        language_dict_features = {
            'hidden': x.view(B, C, -1).transpose(1, 2),
            'masks': torch.ones(B, H*W, device=x.device)
        }
        
        input_dict = {
            'visual': visual_features,
            'lang': language_dict_features
        }
        
        output_dict = self.vl_fuse(input_dict)
        
        fused_visual = output_dict['visual'][0]
        x = fused_visual.view(fused_visual.size(0), -1)
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
