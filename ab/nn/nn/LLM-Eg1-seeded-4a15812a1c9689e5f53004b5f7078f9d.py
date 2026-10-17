import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class DualPathSkipBlock(nn.Module):
    """Dual-path skip block combining convolution and dilated convolutions."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dil1 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h1 = self.act(self.conv1(x))
        h2 = self.act(self.dil1(h1))
        return h2 + x


class HierarchicalFeatureAggregation(nn.Module):
    """Hierarchical feature aggregation using pixel unshuffle and shuffle operations."""

    def __init__(self, channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.unshuffle = nn.PixelUnshuffle(scale_factor)
        self.shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        h = self.unshuffle(x)
        h = self.shuffle(h)
        return h


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 24

        self.in_conv = nn.Conv2d(3, f, 3, padding=1)
        self.dp_skip1 = DualPathSkipBlock(f)
        self.hfa1 = HierarchicalFeatureAggregation(f, 2)
        self.dp_skip2 = DualPathSkipBlock(f)
        self.hfa2 = HierarchicalFeatureAggregation(f, 2)
        self.dp_skip3 = DualPathSkipBlock(f)
        self.out_conv = nn.Conv2d(f, 3, 3, padding=1)

        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(self.device)

    def forward(self, x):
        identity = x
        h = self.in_conv(x)
        h = self.dp_skip1(h)
        h = self.hfa1(h)
        h = self.dp_skip2(h)
        h = self.hfa2(h)
        h = self.dp_skip3(h)
        return torch.clamp(self.out_conv(h) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5
        )

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        bn_layers = [m for m in self.modules() if isinstance(
            m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))]
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            bn_state = [(m.running_mean.clone(), m.running_var.clone(),
                         m.num_batches_tracked.clone()) for m in bn_layers]
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            bad = not torch.isfinite(loss)
            if not bad and bn_layers:
                bad = not all(torch.isfinite(m.running_mean).all()
                              and torch.isfinite(m.running_var).all() for m in bn_layers)
            if bad:
                for m, (rm, rv, nb) in zip(bn_layers, bn_state):
                    m.running_mean.copy_(rm)
                    m.running_var.copy_(rv)
                    m.num_batches_tracked.copy_(nb)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
