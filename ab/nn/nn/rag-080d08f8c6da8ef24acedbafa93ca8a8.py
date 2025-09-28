# Auto-generated single-file for PositionalEncodingFourier
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# register_notrace_module
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
import math

# ---- original imports from contributing modules ----

# ---- PositionalEncodingFourier (target) ----
class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the "Attention is all you Need" paper.
    Based on the official XCiT code
        - https://github.com/facebookresearch/xcit/blob/master/xcit.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.eps = 1e-6

    def forward(self, B: int, H: int, W: int):
        device = self.token_projection.weight.device
        dtype = self.token_projection.weight.dtype
        y_embed = torch.arange(1, H + 1, device=device).to(torch.float32).unsqueeze(1).repeat(1, 1, W)
        x_embed = torch.arange(1, W + 1, device=device).to(torch.float32).repeat(1, H, 1)
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.hidden_dim, device=device).to(torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.hidden_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos.to(dtype))
        return pos.repeat(B, 1, 1, 1)  # (B, C, H, W)

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
        self.pos_encoding = PositionalEncodingFourier(hidden_dim=32, dim=32, temperature=10000)
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        pos_encoding = self.pos_encoding(B, H, W)
        x = x + pos_encoding
        x = x.mean(dim=(2, 3))
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
