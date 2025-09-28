# Auto-generated single-file for TwoStreamFusion
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn

# ---- original imports from contributing modules ----
from torch import nn

# ---- TwoStreamFusion (target) ----
class TwoStreamFusion(nn.Module):
    """A general constructor for neural modules fusing two equal sized tensors
    in forward.

    Args:
        mode (str): The mode of fusion. Options are 'add', 'max', 'min',
            'avg', 'concat'.
    """

    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

        if mode == 'add':
            self.fuse_fn = lambda x: torch.stack(x).sum(dim=0)
        elif mode == 'max':
            self.fuse_fn = lambda x: torch.stack(x).max(dim=0).values
        elif mode == 'min':
            self.fuse_fn = lambda x: torch.stack(x).min(dim=0).values
        elif mode == 'avg':
            self.fuse_fn = lambda x: torch.stack(x).mean(dim=0)
        elif mode == 'concat':
            self.fuse_fn = lambda x: torch.cat(x, dim=1)
        else:
            raise NotImplementedError

    def forward(self, x):
        # split the tensor into two halves in the channel dimension
        x = torch.chunk(x, 2, dim=1)
        return self.fuse_fn(x)

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
        
        self.two_stream_fusion = TwoStreamFusion(mode='concat')
        self.classifier = nn.Linear(64 * 4 * 4, self.num_classes)

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
        
        x = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        B, C, H, W = x.shape
        
        x = self.two_stream_fusion(x)
        
        x = x.view(x.size(0), -1)
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
