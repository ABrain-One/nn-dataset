import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr'}


class MultiResolutionBlock(nn.Module):
    """Multi-resolution block combining dilated and simple blocks for enhanced denoising."""

    def __init__(self, channels):
        super().__init__()
        self.dilated = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.simple = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.dilated(x) + self.simple(x)


class HierarchicalSkipConnection(nn.Module):
    """Hierarchical skip connections combining features from multiple resolutions."""

    def __init__(self, low_res_channels, high_res_channels):
        super().__init__()
        self.low_to_high = nn.Conv2d(low_res_channels, high_res_channels, 1)
        self.high_to_low = nn.Conv2d(high_res_channels, low_res_channels, 1)

    def forward(self, low, high):
        low_upscaled = nn.functional.interpolate(low, size=high.shape[-2:], mode='nearest')
        high_downscaled = nn.functional.interpolate(high, size=low.shape[-2:], mode='nearest')
        return low_upscaled + self.low_to_high(high_downscaled) + self.high_to_low(low)


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 32
        self.unshuffle = nn.PixelUnshuffle(2)
        self.in_conv = nn.Conv2d(3 * 4, f, 3, padding=1)
        self.b0 = MultiResolutionBlock(f)
        self.b1 = MultiResolutionBlock(f)
        self.b2 = MultiResolutionBlock(f)
        self.skip1 = HierarchicalSkipConnection(f, f)
        self.b3 = MultiResolutionBlock(f)
        self.skip2 = HierarchicalSkipConnection(f, f)
        self.b4 = MultiResolutionBlock(f)
        self.out_conv = nn.Conv2d(f, 3 * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        h = self.in_conv(self.unshuffle(x))
        h0 = self.b0(h)
        h1 = self.skip1(h, h0)
        h2 = self.b1(h1)
        h3 = self.skip2(h, h2)
        h4 = self.b2(h3)
        h = self.b3(h4)
        h = self.b4(h)
        return torch.clamp(self.shuffle(self.out_conv(h)) + identity, 0.0, 1.0)

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
