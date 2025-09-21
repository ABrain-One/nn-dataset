# Auto-generated single-file for EMAModule
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---- original imports from contributing modules ----

# ---- EMAModule (target) ----
class EMAModule(nn.Module):
    """Expectation Maximization Attention Module used in EMANet.

    Args:
        channels (int): Channels of the whole module.
        num_bases (int): Number of bases.
        num_stages (int): Number of the EM iterations.
    """

    def __init__(self, channels, num_bases, num_stages, momentum):
        super().__init__()
        assert num_stages >= 1, 'num_stages must be at least 1!'
        self.num_bases = num_bases
        self.num_stages = num_stages
        self.momentum = momentum

        bases = torch.zeros(1, channels, self.num_bases)
        bases.normal_(0, math.sqrt(2. / self.num_bases))
        # [1, channels, num_bases]
        bases = F.normalize(bases, dim=1, p=2)
        self.register_buffer('bases', bases)

    def forward(self, feats):
        """Forward function."""
        batch_size, channels, height, width = feats.size()
        # [batch_size, channels, height*width]
        feats = feats.view(batch_size, channels, height * width)
        # [batch_size, channels, num_bases]
        bases = self.bases.repeat(batch_size, 1, 1)

        with torch.no_grad():
            for i in range(self.num_stages):
                # [batch_size, height*width, num_bases]
                attention = torch.einsum('bcn,bck->bnk', feats, bases)
                attention = F.softmax(attention, dim=2)
                # l1 norm
                attention_normed = F.normalize(attention, dim=1, p=1)
                # [batch_size, channels, num_bases]
                bases = torch.einsum('bcn,bnk->bck', feats, attention_normed)
                # l2 norm
                bases = F.normalize(bases, dim=1, p=2)

        feats_recon = torch.einsum('bck,bnk->bcn', bases, attention)
        feats_recon = feats_recon.view(batch_size, channels, height, width)

        if self.training:
            bases = bases.mean(dim=0, keepdim=True)
            bases = reduce_mean(bases)
            # l2 norm
            bases = F.normalize(bases, dim=1, p=2)
            self.bases = (1 -
                          self.momentum) * self.bases + self.momentum * bases

        return feats_recon

def reduce_mean(tensor):
    return tensor.mean(dim=0, keepdim=True)

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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, self.num_classes)
        
        self.ema_module = EMAModule(
            channels=32,
            num_bases=16,
            num_stages=3,
            momentum=0.9
        )

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.ema_module(x)
        x = self.avgpool(x)
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
