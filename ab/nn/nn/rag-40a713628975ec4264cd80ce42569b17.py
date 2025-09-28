# Auto-generated single-file for PatchEmbed
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import List, Tuple
from typing import List
from typing import Tuple

# ---- PatchEmbed (target) ----
class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
            self,
            dim_in=3,
            dim_out=768,
            kernel=(7, 7),
            stride=(4, 4),
            padding=(3, 3),
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x) -> Tuple[torch.Tensor, List[int]]:
        x = self.proj(x)
        # B C H W -> B HW C
        return x.flatten(2).transpose(1, 2), x.shape[-2:]

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        self.patch_embed = PatchEmbed(
            dim_in=32,
            dim_out=64,
            kernel=(4, 4),
            stride=(2, 2),
            padding=(1, 1)
        )
        
        self.classifier = nn.Linear(64, self.num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x, _ = self.patch_embed(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)
    
    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)
