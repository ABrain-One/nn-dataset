import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {'lr'}


class DualPathContextBlock(nn.Module):
    """Dual-path channel-preserving residual block with context."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 2, 3, padding=1)
        self.conv2 = nn.Conv2d(channels // 2, channels, 3, padding=1)
        self.ctx_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.ctx = nn.Conv2d(channels, channels, 5, padding=2)
        self.act = nn.ReLU(inplace=True)
        self.skip_proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        identity = self.skip_proj(x)
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        ctx = self.ctx(self.ctx_pool(h))
        ctx = F.interpolate(ctx, size=h.shape[-2:], mode='nearest')
        return torch.clamp(identity + h + ctx, 0.0, 1.0)


class Net(nn.Module):
    """Combining dual-path blocks with ELU_avg for effective denoising."""

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 32
        self.unshuffle = nn.PixelUnshuffle(2)
        self.in_conv = nn.Conv2d(3 * 4, f, 3, padding=1)
        self.dpab0 = DualPathContextBlock(f)
        self.dpab1 = DualPathContextBlock(f)
        self.dpab2 = DualPathContextBlock(f)
        self.dpab3 = DualPathContextBlock(f)
        self.dpab4 = DualPathContextBlock(f)
        self.dpab5 = DualPathContextBlock(f)
        self.out_conv = nn.Conv2d(f, 3 * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        h = self.in_conv(self.unshuffle(x))
        h = self.dpab0(h)
        h = self.dpab1(h)
        h = self.dpab2(h)
        h = self.dpab3(h)
        h = self.dpab4(h)
        h = self.dpab5(h)
        return torch.clamp(self.shuffle(self.out_conv(h)) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=prm.get("epoch_max", 50), eta_min=1e-5
        )

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
