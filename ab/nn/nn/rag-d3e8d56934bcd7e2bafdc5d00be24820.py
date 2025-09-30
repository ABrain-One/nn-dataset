# Auto-generated single-file for Pool
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import Optional
from typing import TypeVar
T = TypeVar("T")

# ---- torchvision.models.video.mvit._squeeze ----
def _squeeze(x: torch.Tensor, target_dim: int, expand_dim: int, tensor_dim: int) -> torch.Tensor:
    if tensor_dim == target_dim - 1:
        x = x.squeeze(expand_dim)
    return x

# ---- torchvision.models.video.mvit._unsqueeze ----
def _unsqueeze(x: torch.Tensor, target_dim: int, expand_dim: int) -> tuple[torch.Tensor, int]:
    tensor_dim = x.dim()
    if tensor_dim == target_dim - 1:
        x = x.unsqueeze(expand_dim)
    elif tensor_dim != target_dim:
        raise ValueError(f"Unsupported input dimension {x.shape}")
    return x, tensor_dim

# ---- Pool (target) ----
class Pool(nn.Module):
    def __init__(
        self,
        pool: nn.Module,
        norm: Optional[nn.Module],
        activation: Optional[nn.Module] = None,
        norm_before_pool: bool = False,
    ) -> None:
        super().__init__()
        self.pool = pool
        layers = []
        if norm is not None:
            layers.append(norm)
        if activation is not None:
            layers.append(activation)
        self.norm_act = nn.Sequential(*layers) if layers else None
        self.norm_before_pool = norm_before_pool

    def forward(self, x: torch.Tensor, thw: tuple[int, int, int]) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x, tensor_dim = _unsqueeze(x, 4, 1)

        # Separate the class token and reshape the input
        class_token, x = torch.tensor_split(x, indices=(1,), dim=2)
        x = x.transpose(2, 3)
        B, N, C = x.shape[:3]
        x = x.reshape((B * N, C) + thw).contiguous()

        # normalizing prior pooling is useful when we use BN which can be absorbed to speed up inference
        if self.norm_before_pool and self.norm_act is not None:
            x = self.norm_act(x)

        # apply the pool on the input and add back the token
        x = self.pool(x)
        T, H, W = x.shape[2:]
        x = x.reshape(B, N, C, -1).transpose(2, 3)
        x = torch.cat((class_token, x), dim=2)

        if not self.norm_before_pool and self.norm_act is not None:
            x = self.norm_act(x)

        x = _squeeze(x, 4, 1, tensor_dim)
        return x, (T, H, W)

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
        self.pool = Pool(
            pool=nn.AdaptiveAvgPool2d((1, 1)),
            norm=nn.BatchNorm2d(32),
            activation=nn.ReLU(),
            norm_before_pool=False
        )
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.mean(dim=(2, 3))
        return self.classifier(x)

    def train_setup(self, prm):
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
