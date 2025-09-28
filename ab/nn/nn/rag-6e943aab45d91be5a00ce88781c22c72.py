# Auto-generated single-file for PosEmbedRel
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import collections
from itertools import repeat
from collections import *

# ---- timm.layers.bottleneck_attn.rel_logits_1d ----
def rel_logits_1d(q, rel_k, permute_mask: List[int]):
    """ Compute relative logits along one dimension

    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925

    Args:
        q: (batch, heads, height, width, dim)
        rel_k: (2 * width - 1, dim)
        permute_mask: permute output dim according to this
    """
    B, H, W, dim = q.shape
    x = (q @ rel_k.transpose(-1, -2))
    x = x.reshape(-1, W, 2 * W -1)

    # pad to shift from relative to absolute indexing
    x_pad = F.pad(x, [0, 1]).flatten(1)
    x_pad = F.pad(x_pad, [0, W - 1])

    # reshape and slice out the padded elements
    x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    x = x_pad[:, :W, W - 1:]

    # reshape and tile
    x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    return x.permute(permute_mask)

# ---- timm.layers.helpers._ntuple ----
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

# ---- timm.layers.helpers.to_2tuple ----
to_2tuple = _ntuple(2)

# ---- PosEmbedRel (target) ----
class PosEmbedRel(nn.Module):
    """ Relative Position Embedding
    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925
    """
    def __init__(self, feat_size, dim_head, scale):
        super().__init__()
        self.height, self.width = to_2tuple(feat_size)
        self.dim_head = dim_head
        self.height_rel = nn.Parameter(torch.randn(self.height * 2 - 1, dim_head) * scale)
        self.width_rel = nn.Parameter(torch.randn(self.width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        B, HW, _ = q.shape

        # relative logits in width dimension.
        q = q.reshape(B, self.height, self.width, -1)
        rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))

        # relative logits in height dimension.
        q = q.transpose(1, 2)
        rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))

        rel_logits = rel_logits_h + rel_logits_w
        rel_logits = rel_logits.reshape(B, HW, HW)
        return rel_logits

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
        self.pos_embed_rel = PosEmbedRel(feat_size=8, dim_head=4, scale=0.02)
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
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((8, 8))
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        if H != 8 or W != 8:
            x = F.adaptive_avg_pool2d(x, (8, 8))
            B, C, H, W = x.shape
        
        x_3d = x.flatten(2).transpose(1, 2)
        
        rel_pos_features = self._create_relative_position_features(x_3d)
        x = x_3d.mean(dim=1) + rel_pos_features.mean(dim=1)
        return self.classifier(x)
    
    def _create_relative_position_features(self, x):
        """Create simple relative position features"""
        B, N, C = x.shape
        pos = torch.arange(N, device=x.device).float()
        pos = pos.unsqueeze(0).unsqueeze(-1)
        pos = pos.expand(B, N, C)  
        pos_encoding = torch.sin(pos / 10000.0) * 0.1
        return pos_encoding

    def train_setup(self, prm):
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
