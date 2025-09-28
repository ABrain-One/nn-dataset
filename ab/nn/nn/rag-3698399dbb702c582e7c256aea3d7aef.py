# Auto-generated single-file for EmbeddingEMA
# Dependencies are emitted in topological order (utilities first).
def rearrange(tensor, pattern):
    """Simple rearrange function for tensor reshaping"""
    if pattern == 'n d -> n () d':
        return tensor.unsqueeze(1)
    elif pattern == 'c d -> () c d':
        return tensor.unsqueeze(0)
    else:
        return tensor

# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from itertools import repeat
from typing import Optional
from typing import Tuple

# ---- mmpretrain.models.utils.vector_quantizer.sample_vectors ----
def sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    """Sample vectors according to the given number."""
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num, ), device=device)

    return samples[indices]

# ---- mmpretrain.models.utils.vector_quantizer.kmeans ----
def kmeans(samples: torch.Tensor,
           num_clusters: int,
           num_iters: int = 10,
           use_cosine_sim: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run k-means algorithm."""
    dim, dtype, _ = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = F.normalize(new_means, p=2, dim=-1)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins

# ---- EmbeddingEMA (target) ----
class EmbeddingEMA(nn.Module):
    """The codebook of embedding vectors.

    Args:
        num_tokens (int): Number of embedding vectors in the codebook.
        codebook_dim (int) : The dimension of embedding vectors in the
            codebook.
        kmeans_init (bool): Whether to use k-means to initialize the
            VectorQuantizer. Defaults to True.
        codebook_init_path (str): The initialization checkpoint for codebook.
            Defaults to None.
    """

    def __init__(self,
                 num_tokens: int,
                 codebook_dim: int,
                 kmeans_init: bool = True,
                 codebook_init_path: Optional[str] = None):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        if codebook_init_path is None:
            if not kmeans_init:
                weight = torch.randn(num_tokens, codebook_dim)
                weight = F.normalize(weight, p=2, dim=-1)
            else:
                weight = torch.zeros(num_tokens, codebook_dim)
            self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        else:
            print(f'load init codebook weight from {codebook_init_path}')
            codebook_ckpt_weight = torch.load(
                codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', torch.Tensor([True]))

        self.weight = nn.Parameter(weight, requires_grad=True)
        self.update = True

    def init_embed_(self, data: torch.Tensor) -> None:
        """Initialize embedding vectors of codebook."""
        if self.initted:
            return
        print('Performing K-means init for codebook')
        embed, _ = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.weight.data.copy_(embed)
        self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, embed_id: torch.Tensor) -> torch.Tensor:
        """Get embedding vectors."""
        return F.embedding(embed_id, self.weight)

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
        self.projection = nn.Linear(32, 32)
        self.classifier = nn.Linear(1000, self.num_classes)
        self.embedding_ema = EmbeddingEMA(num_tokens=32, codebook_dim=1000, kmeans_init=False)

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
        
        x_proj = self.projection(x)
        x_embedded = torch.nn.functional.linear(x_proj, self.embedding_ema.weight.t())
        
        return self.classifier(x_embedded)

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
