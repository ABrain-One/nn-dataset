import torch
import torch.nn as nn
import torch.optim as optim

class PBlock(nn.Module):
    """Grouped conv + ChannelShuffle + Hardswish residual block."""

    def __init__(self, channels):
        super().__init__()
        g = 4 if channels % 4 == 0 else 1
        self.conv1 = nn.Conv2d(channels, channels, 1, padding=0, groups=g)
        self.shuffle = nn.ChannelShuffle(g)
        self.conv2 = nn.Conv2d(channels, channels, 7, padding=3)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        h = self.act(self.conv1(x))
        return self.conv2(self.shuffle(h)) + x

class Net(nn.Module):
    """PARALLEL MULTI-RESOLUTION (HRNet-style) denoiser. A full-resolution branch and a
    half-resolution branch run SIMULTANEOUSLY through every stage, exchanging information
    after each one. Not an encoder->decoder (no bottleneck) and not a flat stack: the graph
    has two persistent parallel branches with repeated bidirectional fusion."""

    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f1, f2 = (12, 24)
        self.stem = nn.Conv2d(3, f1, 3, padding=1)
        self.down0 = nn.Conv2d(f1, f2, 3, stride=2, padding=1)
        self.h1 = PBlock(f1)
        self.h2 = PBlock(f1)
        self.h3 = PBlock(f1)
        self.l1 = PBlock(f2)
        self.l2 = PBlock(f2)
        self.l3 = PBlock(f2)
        self.lo2hi_1 = nn.Conv2d(f2, f1, 1)
        self.lo2hi_2 = nn.Conv2d(f2, f1, 1)
        self.hi2lo_1 = nn.Conv2d(f1, f2, 5, stride=2, padding=2)
        self.hi2lo_2 = nn.Conv2d(f1, f2, 3, stride=2, padding=1)
        self.lo_proj = nn.Conv2d(f2, f1, 1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.head = nn.Conv2d(f1 * 2, 3, 5, padding=2)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        hi = self.stem(x)
        lo = self.down0(hi)
        hi = self.h1(hi)
        lo = self.l1(lo)
        hi = hi + self.up(self.lo2hi_1(lo))
        lo = lo + self.hi2lo_1(hi)
        hi = self.h2(hi)
        lo = self.l2(lo)
        hi = hi + self.up(self.lo2hi_2(lo))
        lo = lo + self.hi2lo_2(hi)
        hi = self.h3(hi)
        lo = self.l3(lo)
        out = self.head(torch.cat([hi, self.up(self.lo_proj(lo))], dim=1))
        return torch.clamp(out + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        bn_layers = [m for m in self.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))]
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
            self.optimizer.zero_grad()
            bn_state = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone()) for m in bn_layers]
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            bad = not torch.isfinite(loss)
            if not bad and bn_layers:
                bad = not all((torch.isfinite(m.running_mean).all() and torch.isfinite(m.running_var).all() for m in bn_layers))
            if bad:
                for m, (rm, rv, nb) in zip(bn_layers, bn_state):
                    m.running_mean.copy_(rm)
                    m.running_var.copy_(rv)
                    m.num_batches_tracked.copy_(nb)
                continue
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)