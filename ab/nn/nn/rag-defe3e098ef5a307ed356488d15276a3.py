# Auto-generated single-file for GatedCrossAttentionBlock
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# einsum, rearrange
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
from typing import Optional
from itertools import repeat

# Add missing functions
def einsum(equation, *operands):
    return torch.einsum(equation, *operands)

def rearrange(tensor, pattern, **kwargs):
    # Simple implementation for common patterns
    if pattern == 'b t n d -> b (t n) d':
        b, t, n, d = tensor.shape
        return tensor.view(b, t * n, d)
    elif pattern == 'b n (h d) -> b h n d':
        b, n, hd = tensor.shape
        h = kwargs.get('h', 8)  # default to 8 heads
        d = hd // h
        return tensor.view(b, h, n, d)
    elif pattern == 'b h n d -> b n (h d)':
        b, h, n, d = tensor.shape
        return tensor.view(b, n, h * d)
    elif pattern == 'b i -> b 1 i 1':
        b, i = tensor.shape
        return tensor.unsqueeze(1).unsqueeze(-1)
    elif pattern == 'j -> 1 1 1 (j n)':
        j = tensor.shape[0]
        n = kwargs.get('n', 1)
        return tensor.view(1, 1, 1, j * n)
    elif pattern == 'b i -> b 1 i 1':
        b, i = tensor.shape
        return tensor.unsqueeze(1).unsqueeze(-1)
    else:
        return tensor

# ---- mmpretrain.models.multimodal.flamingo.modules.MaskedCrossAttention ----
class MaskedCrossAttention(nn.Module):
    """Masked cross attention layers.

    Args:
        dim (int): Input text feature dimensions.
        dim_visual (int): Input visual feature dimensions.
        dim_head (int): Number of dimension heads. Defaults to 64.
        heads (int): Number of heads. Defaults to 8.
        only_attend_immediate_media (bool): Whether attend immediate media.
            Defaults to True.
    """

    def __init__(
        self,
        *,
        dim: int,
        dim_visual: int,
        dim_head: int = 64,
        heads: int = 8,
        only_attend_immediate_media: bool = True,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image
        # or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self,
                x: torch.Tensor,
                media: torch.Tensor,
                media_locations: Optional[torch.Tensor] = None,
                attend_previous: bool = True):
        """Forward function for perceiver sampler.

        Args:
            x (torch.Tensor): text features of shape (B, T_txt, D_txt).
            media (torch.Tensor): image features of shape
                (B, T_img, n, D_img) where n is the dim of the latents.
            media_locations (torch.Tensor, optional): boolean mask identifying
                the media tokens in x of shape (B, T_txt). Defaults to None.
            attend_previous (bool): If false, ignores immediately preceding
                image and starts attending when following image.
                Defaults to True.
        """
        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, 'b t n d -> b (t n) d')

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)

        if media_locations is not None:
            # at each boolean of True, increment the time counter
            # (relative to media time)
            text_time = media_locations.cumsum(dim=-1)
            media_time = torch.arange(T_img, device=x.device) + 1

            if not attend_previous:
                text_time[~media_locations] += 1
                # make sure max is still the number of images in the sequence
                text_time[text_time > repeat(
                    torch.count_nonzero(media_locations, dim=1),
                    'b -> b i',
                    i=text_time.shape[1],
                )] = 0

            # text time must equal media time if only attending to most
            # immediate image otherwise, as long as text time is greater than
            # media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge  # noqa

            text_to_media_mask = mask_op(
                rearrange(text_time, 'b i -> b 1 i 1'),
                repeat(media_time, 'j -> 1 1 1 (j n)', n=n),
            )
            sim = sim.masked_fill(~text_to_media_mask,
                                  -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if media_locations is not None and self.only_attend_immediate_media:
            # any text without a preceding media needs to have
            # attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask,
                                                'b i -> b 1 i 1')
            attn = attn.masked_fill(text_without_media_mask, 0.0)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# ---- mmpretrain.models.multimodal.flamingo.modules.FeedForward ----
def FeedForward(dim, mult: int = 4):
    """Feedforward layers.

    Args:
        mult (int): Layer expansion muliplier. Defaults to 4.
    """
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

# ---- GatedCrossAttentionBlock (target) ----
class GatedCrossAttentionBlock(nn.Module):
    """Gated cross attention layers.

    Args:
        dim (int): Input text feature dimensions.
        dim_visual (int): Input visual feature dimensions.
        dim_head (int): Number of dimension heads. Defaults to 64.
        heads (int): Number of heads. Defaults to 8.
        ff_mult (int): Feed forward multiplier. Defaults to 4.
        only_attend_immediate_media (bool): Whether attend immediate media.
            Defaults to True.
    """

    def __init__(
        self,
        *,
        dim: int,
        dim_visual: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        only_attend_immediate_media: bool = True,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self,
                x: torch.Tensor,
                media: torch.Tensor,
                media_locations: Optional[torch.Tensor] = None,
                attend_previous: bool = True):
        """Forward function for perceiver sampler.

        Args:
            x (torch.Tensor): text features of shape (B, T_txt, D_txt).
            media (torch.Tensor): image features of shape
                (B, T_img, n, D_img) where n is the dim of the latents.
            media_locations (torch.Tensor, optional): boolean mask identifying
                the media tokens in x of shape (B, T_txt). Defaults to None.
            attend_previous (bool): If false, ignores immediately preceding
                image and starts attending when following image.
                Defaults to True.
        """
        x = (
            self.attn(
                x,
                media,
                media_locations=media_locations,
                attend_previous=attend_previous,
            ) * self.attn_gate.tanh() + x)
        x = self.ff(x) * self.ff_gate.tanh() + x

        return x

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
        self.gated_cross_attn = GatedCrossAttentionBlock(dim=32, dim_visual=32, dim_head=8, heads=4, ff_mult=4)
        self.classifier = nn.Linear(32, self.num_classes)

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
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        media = torch.randn(batch_size, 1, 1, 32, device=x.device)
        x = self.gated_cross_attn(x, media)
        x = x.squeeze(1)
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
