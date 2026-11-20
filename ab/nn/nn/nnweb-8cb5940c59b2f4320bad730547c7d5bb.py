import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation, MLP


class MLPBlock(MLP):
    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i + 1}.{type}"
                    new_key = f"{prefix}{3 * i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    def __init__(
            self,
            seq_length: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'patch_size'}


def get_divisors(n, res=None):
    res = res or []
    i = 1
    while i <= n:
        if n % i == 0:
            res.append(i)
        i = i + 1
    return res


def get_closest_split(n, close_to):
    all_divisors = get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if close_to < val:
            if ix == 0: return val
            if (val - close_to) > (close_to - all_divisors[ix - 1]):
                return all_divisors[ix - 1]
            return val


class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        image_size: int = in_shape[2]
        
        # Use the patch_size hyperparameter to determine the stem stride.
        # For CIFAR-10 (32x32), if patch_size is ~0.1-0.2, we get a stride of 4 or 8.
        # We ensure the stride is a divisor of the image size.
        # We want a small stride to keep spatial resolution high for classification.
        # Let's aim for a stride that results in a reasonable number of tokens.
        # If patch_size is small, we use a larger stride to reduce sequence length for efficiency, 
        # but for CIFAR we want high resolution. 
        # Actually, let's just use the patch_size to define the initial conv kernel/stride.
        # Standard ViT uses 16x16 for 224x224. For 32x32, 4x4 or 8x8 is common.
        # Let's derive a stride from patch_size.
        # If patch_size is 0.1, 32*0.1 = 3.2 -> closest divisor 4.
        # If patch_size is 0.2, 32*0.2 = 6.4 -> closest divisor 8? Or 4?
        # get_closest_split(32, 6.4) -> divisors: 1,2,4,8,16,32. 
        # 6.4 is between 4 and 8. |6.4-4|=2.4, |6.4-8|=1.6. So 8.
        
        target_patch = int(image_size * prm['patch_size'])
        # Ensure we don't go below 1
        target_patch = max(1, target_patch)
        stride = get_closest_split(image_size, target_patch)
        
        # We also need to ensure the resulting feature map size is reasonable.
        # Let's cap the stride at 8 to avoid too few tokens for small images, 
        # but respect the divisor constraint.
        # Actually, for CIFAR-10, a stride of 4 (8x8 tokens) or 8 (4x4 tokens) is typical.
        # A stride of 4 gives 64 tokens. A stride of 8 gives 16 tokens.
        # Let's stick with the calculated stride.
        
        num_classes: int = out_shape[0]
        
        # Architecture: EfficientViT style or a robust ViT with Conv Stem
        # We will use a Conv Stem to project to hidden_dim, then ViT blocks.
        # We will use a smaller ViT (Base-S) to be efficient but effective.
        # Hidden dim: 384, Heads: 6, Layers: 12, MLP: 1536
        # This is roughly 30M params, good for CIFAR.
        
        hidden_dim = 384
        num_heads = 6
        num_layers = 12
        mlp_dim = 1536
        
        dropout = prm['dropout']
        attention_dropout = prm['attention_dropout']
        
        # Conv Stem
        # Input: 3 x H x W
        # Output: hidden_dim x (H/stride) x (W/stride)
        # We use a single conv layer with kernel_size=stride, stride=stride
        # This is efficient and standard.
        
        self.conv_proj = nn.Conv2d(
            in_channels=in_shape[1], 
            out_channels=hidden_dim, 
            kernel_size=stride, 
            stride=stride
        )
        
        # Initialize Conv Stem
        fan_in = in_shape[1] * stride * stride
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)
        
        # Calculate sequence length
        feature_h = image_size // stride
        feature_w = image_size // stride
        seq_length = feature_h * feature_w
        
        # Class Token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1 # include class token
        
        # Position Embedding
        # We use learnable position embeddings.
        # For better generalization, we could use sincos, but learnable is fine for CIFAR.
        # We add a positional embedding to the tokens.
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        # ViT Encoder
        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        
        # Head
        # We use a simple linear head.
        self.head = nn.Linear(hidden_dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        
        # Store hyperparams for potential use or debugging
        self.image_size = image_size
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        # Resize if necessary? The problem says "Never hard-code the input resolution; derive what you need from in_shape."
        # It implies the input might vary or we should handle it generically.
        # However, CIFAR-10 is fixed 32x32.
        # If input size differs from self.image_size, we should resize.
        if h != self.image_size or w != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        x = self.conv_proj(x)
        # x shape: (N, C, H', W')
        n, c, h_p, w_p = x.shape
        x = x.reshape(n, c, h_p * w_p)
        x = x.permute(0, 2, 1) # (N, L, C)
        return x

    def forward(self, x: torch.Tensor):
        x = self._process_input(x)
        n = x.shape[0]
        
        # Prepend class token
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Encode
        x = self.encoder(x)
        
        # Take class token output
        x = x[:, 0]
        
        # Classification
        x = self.head(x)
        return x