# Auto-generated single-file for GaussianFilter
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
from torch import Tensor

# ---- original imports from contributing modules ----

# ---- GaussianFilter (target) ----
class GaussianFilter(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (float(offset[1]) - float(offset[0]))**2
        self.register_buffer('offset', offset)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * dist.pow(2))

def supported_hyperparameters():
    return {'lr', 'momentum'}

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
        self.gaussian_filter = GaussianFilter(start=0.0, stop=5.0, num_gaussians=50)
        self.classifier = torch.nn.Linear(50, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=False),
        ]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        
        # Create distance-like features for GaussianFilter
        batch_size = x.size(0)
        dist_features = torch.norm(x, dim=1, keepdim=True)
        gaussian_features = self.gaussian_filter(dist_features)
        gaussian_features = gaussian_features.squeeze(1)
        
        return self.classifier(gaussian_features)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
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
