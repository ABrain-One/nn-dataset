# Auto-generated single-file for Affine
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn

# ---- original imports from contributing modules ----

# ---- Affine (target) ----
class Affine(nn.Module):
    """Affine transformation layer."""

    def __init__(self, dim: int) -> None:
        """Initialize Affine layer.

        Args:
            dim: Dimension of features.
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply affine transformation."""
        return torch.addcmul(self.beta, self.alpha, x)


def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        layers += [
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        class AffineWrapper(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
                self.bn = nn.BatchNorm2d(out_channels)
                self.affine = Affine(out_channels)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                B, C, H, W = x.shape
                x = x.permute(0, 2, 3, 1).contiguous()
                x = x.view(B, H*W, C)
                x = self.affine(x)
                x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                return x
        
        layers += [
            AffineWrapper(32, 32),
            nn.ReLU(inplace=True),
        ]

        self._last_channels = 32
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
