# Auto-generated single-file for _MLPMixer
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# Linear
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm
import torch.nn as nn

# ---- original imports from contributing modules ----
from torch.nn import LayerNorm, Linear

# ---- _MLPMixer (target) ----
class _MLPMixer(torch.nn.Module):
    r"""The MLP-Mixer module.

    Args:
        num_tokens (int): Number of tokens/patches in each sample.
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
    """
    def __init__(
        self,
        num_tokens: int,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dropout = dropout

        self.token_norm = LayerNorm(in_channels)
        self.token_lin1 = Linear(num_tokens, num_tokens // 2)
        self.token_lin2 = Linear(num_tokens // 2, num_tokens)

        self.channel_norm = LayerNorm(in_channels)
        self.channel_lin1 = Linear(in_channels, 4 * in_channels)
        self.channel_lin2 = Linear(4 * in_channels, in_channels)

        self.head_norm = LayerNorm(in_channels)
        self.head_lin = Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.token_norm.reset_parameters()
        self.token_lin1.reset_parameters()
        self.token_lin2.reset_parameters()
        self.channel_norm.reset_parameters()
        self.channel_lin1.reset_parameters()
        self.channel_lin2.reset_parameters()
        self.head_norm.reset_parameters()
        self.head_lin.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Tensor of size
                :obj:`[*, num_tokens, in_channels]`.

        Returns:
            Tensor of size :obj:`[*, out_channels]`.
        """
        # Token mixing:
        h = self.token_norm(x).mT
        h = self.token_lin1(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.token_lin2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h_token = h.mT + x

        # Channel mixing:
        h = self.channel_norm(h_token)
        h = self.channel_lin1(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.channel_lin2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h_channel = h + h_token

        # Head:
        out = self.head_norm(h_channel)
        out = out.mean(dim=1)
        out = self.head_lin(out)
        return out

def supported_hyperparameters():
    return ['lr', 'momentum']

class Net(torch.nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        
        self.mlp_mixer = _MLPMixer(
            num_tokens=16,
            in_channels=64,
            out_channels=self.num_classes,
            dropout=0.0
        )

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
        
        x = F.adaptive_avg_pool2d(x, (4, 4))
        B, C, H, W = x.shape
        
        x = x.view(B, C, H * W).transpose(1, 2)
        
        x = self.mlp_mixer(x)
        
        return x

    def train_setup(self, prm: dict):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = torch.nn.CrossEntropyLoss()

    def learn(self, data_roll):
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()       
