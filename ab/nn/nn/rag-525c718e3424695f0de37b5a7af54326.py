# Auto-generated single-file for RelativePositionBias
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ---- timm.layers.grid.ndgrid ----
def ndgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    This naming is based on MATLAB, the purpose is to avoid confusion due to torch's change to make
    torch.meshgrid behaviour move from matching ndgrid ('ij') indexing to numpy meshgrid defaults of ('xy').

    """
    try:
        return torch.meshgrid(*tensors, indexing='ij')
    except TypeError:
        # old PyTorch < 1.10 will follow this path as it does not have indexing arg,
        # the old behaviour of meshgrid was 'ij'
        return torch.meshgrid(*tensors)

# ---- timm.models.beit.gen_relative_position_index ----
def gen_relative_position_index(window_size: Tuple[int, int]) -> torch.Tensor:
    """Generate relative position index for window-based attention.

    Creates a lookup table for relative position indices between all pairs of positions
    within a window, including special handling for cls token interactions.

    Args:
        window_size: Height and width of the attention window.

    Returns:
        Relative position index tensor of shape (window_area+1, window_area+1)
        where +1 accounts for the cls token.
    """
    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
    # cls to token & token 2 cls & cls to cls
    # get pair-wise relative position index for each token inside the window
    window_area = window_size[0] * window_size[1]
    coords = torch.stack(ndgrid(torch.arange(window_size[0]), torch.arange(window_size[1])))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = torch.zeros(size=(window_area + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index

# ---- RelativePositionBias (target) ----
class RelativePositionBias(nn.Module):
    """Relative position bias module for window-based attention.

    Generates learnable relative position biases for all pairs of positions
    within a window, including special handling for cls token.
    """

    def __init__(self, window_size: Tuple[int, int], num_heads: int):
        """Initialize relative position bias module.

        Args:
            window_size: Height and width of the attention window.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_distance, num_heads))
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.register_buffer("relative_position_index", gen_relative_position_index(window_size))

    def forward(self) -> torch.Tensor:
        """Generate relative position bias.

        Returns:
            Relative position bias tensor of shape (num_heads, window_area+1, window_area+1).
        """
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_area + 1, self.window_area + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

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
        self.rel_pos_bias = RelativePositionBias(window_size=(4, 4), num_heads=8)
        self.classifier = nn.Linear(64, self.num_classes)

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
        x = F.adaptive_avg_pool2d(x, (4, 4))
        B, C, H, W = x.shape
        rel_pos_bias = self.rel_pos_bias()
        rel_pos_bias = rel_pos_bias[:, 1:, 1:]  
        attn = torch.randn(B, 8, H * W, H * W, device=x.device)
        attn = attn + rel_pos_bias
        x = x.view(B, C, -1).mean(dim=2)
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
