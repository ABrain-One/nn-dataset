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
    return {'lr','momentum'}


class Net(torch.nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
        # Stable 2D stem to avoid channel/shape mismatches
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]

        layers += [
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]

        # Use the provided _MLPMixer block at least once.
        # Create a simpler adapter that uses MLPMixer as a feature extractor
        class MLPMixerAdapter(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.mlp_mixer = _MLPMixer(
                    num_tokens=1,  # Single token after global pooling
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout=0.1
                )
                self.spatial_proj = torch.nn.Conv2d(out_channels, out_channels, 1)
            
            def forward(self, x):
                B, C, H, W = x.shape
                x_pooled = self.global_pool(x)
                x_flat = x_pooled.flatten(2).transpose(1, 2)
                x_out = self.mlp_mixer(x_flat)
                x_out = x_out.unsqueeze(-1).unsqueeze(-1)
                x_out = self.spatial_proj(x_out)
                x_out = torch.nn.functional.interpolate(x_out, size=(H, W), mode='bilinear', align_corners=False)
                return x_out
        
        layers += [
            MLPMixerAdapter(32, 32),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]
        self._last_channels = 32
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
