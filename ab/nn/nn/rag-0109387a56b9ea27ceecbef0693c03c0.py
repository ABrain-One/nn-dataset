# Auto-generated single-file for PositionalEncoding
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import Optional

# ---- original imports from contributing modules ----

# ---- PositionalEncoding (target) ----
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, spatial_size: tuple[int, int], temporal_size: int, rel_pos_embed: bool) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.temporal_size = temporal_size

        self.class_token = nn.Parameter(torch.zeros(embed_size))
        self.spatial_pos: Optional[nn.Parameter] = None
        self.temporal_pos: Optional[nn.Parameter] = None
        self.class_pos: Optional[nn.Parameter] = None
        if not rel_pos_embed:
            self.spatial_pos = nn.Parameter(torch.zeros(self.spatial_size[0] * self.spatial_size[1], embed_size))
            self.temporal_pos = nn.Parameter(torch.zeros(self.temporal_size, embed_size))
            self.class_pos = nn.Parameter(torch.zeros(embed_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        class_token = self.class_token.expand(x.size(0), -1).unsqueeze(1)
        x = torch.cat((class_token, x), dim=1)

        if self.spatial_pos is not None and self.temporal_pos is not None and self.class_pos is not None:
            hw_size, embed_size = self.spatial_pos.shape
            pos_embedding = torch.repeat_interleave(self.temporal_pos, hw_size, dim=0)
            pos_embedding.add_(self.spatial_pos.unsqueeze(0).expand(self.temporal_size, -1, -1).reshape(-1, embed_size))
            pos_embedding = torch.cat((self.class_pos.unsqueeze(0), pos_embedding), dim=0).unsqueeze(0)
            x.add_(pos_embedding)

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
        self.pos_encoding = PositionalEncoding(embed_size=32, spatial_size=(4, 4), temporal_size=1, rel_pos_embed=False)
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
        x = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_encoding(x)
        x = x.mean(dim=1)
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
