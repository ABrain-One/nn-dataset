# Auto-generated single-file for GlobalLocalFilter
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- mmpretrain.models.backbones.hornet.HorNetLayerNorm ----
class HorNetLayerNorm(nn.Module):
    """An implementation of LayerNorm of HorNet.

    The differences between HorNetLayerNorm & torch LayerNorm:
        1. Supports two data formats channels_last or channels_first.
    Args:
        normalized_shape (int or list or torch.Size): input shape from an
            expected input of size.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        data_format (str): The ordering of the dimensions in the inputs.
            channels_last corresponds to inputs with shape (batch_size, height,
            width, channels) while channels_first corresponds to inputs with
            shape (batch_size, channels, height, width).
            Defaults to 'channels_last'.
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise ValueError(
                'data_format must be channels_last or channels_first')
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# ---- GlobalLocalFilter (target) ----
class GlobalLocalFilter(nn.Module):
    """A GlobalLocalFilter of HorNet.

    Args:
        dim (int): Number of input channels.
        h (int): Height of complex_weight.
            Defaults to 14.
        w (int): Width of complex_weight.
            Defaults to 8.
    """

    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(
            dim // 2,
            dim // 2,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=dim // 2)
        self.complex_weight = nn.Parameter(
            torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        self.pre_norm = HorNetLayerNorm(
            dim, eps=1e-6, data_format='channels_first')
        self.post_norm = HorNetLayerNorm(
            dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(
                weight.permute(3, 0, 1, 2),
                size=x2.shape[2:4],
                mode='bilinear',
                align_corners=True).permute(1, 2, 3, 0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)],
                      dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
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
        self.global_local_filter = GlobalLocalFilter(dim=32, h=8, w=8)
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.global_local_filter(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
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
