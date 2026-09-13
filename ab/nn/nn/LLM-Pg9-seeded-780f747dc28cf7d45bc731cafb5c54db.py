import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {'lr', 'batch', 'epoch_max', 'transform'}

class MultiPathAttnBlock(nn.Module):
    """LPPool2d context branch + Hardsigmoid gate."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.pool = nn.LPPool2d(2, 4)
        self.ctx = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Hardsigmoid(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        h = self.conv1(x)
        c = self.ctx(self.pool(h))
        c = F.interpolate(c, size=h.shape[-2:], mode='nearest')
        return self.conv2(h * self.gate(c)) + x

class GroupedConvBlock(nn.Module):
    """Grouped conv + Hardswish residual block."""

    def __init__(self, channels):
        super().__init__()
        g = 4 if channels % 4 == 0 else 1
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, groups=g)
        self.shuffle = nn.ChannelShuffle(g)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        h = self.act(self.conv1(x))
        return self.conv2(self.shuffle(h)) + x

class SplitNode(nn.Module):
    """One encoder/decoder node: split channels, conv each half down, recurse (or mid at leaf),
    concat siblings, transpose up, add skip. Distinct weights per node => true multi-path tree."""

    def __init__(self, f, level, depth):
        super().__init__()
        self.half = f // 2
        self.pad = nn.ReflectionPad2d(1)
        self.conv_a = nn.Conv2d(self.half, f, 3, stride=2)
        self.conv_b = nn.Conv2d(self.half, f, 3, stride=2)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.leaf = level >= depth - 1
        if self.leaf:
            self.mid_a = MultiPathAttnBlock(f)
            self.mid_b = MultiPathAttnBlock(f)
        else:
            self.child_a = SplitNode(f, level + 1, depth)
            self.child_b = SplitNode(f, level + 1, depth)
        self.up = nn.ConvTranspose2d(2 * f, f, 2, stride=2)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        skip = x
        a = self.act(self.conv_a(self.pad(x[:, :self.half])))
        b = self.act(self.conv_b(self.pad(x[:, self.half:])))
        if self.leaf:
            a = self.mid_a(a); b = self.mid_b(b)
        else:
            a = self.child_a(a); b = self.child_b(b)
        m = self.act2(self.up(torch.cat([a, b], dim=1)))
        return m + skip

class Net(nn.Module):
    """Combined denoising architecture using attention, channel-shuffling, and hybrid block."""

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f0 = 16
        f1, f2, f3, f4 = (f0 * 2, f0 * 4, f0 * 8, f0 * 16)
        self.in_conv = nn.Conv2d(3, f0, 3, padding=1)
        self.eb0 = GroupedConvBlock(f0)
        self.down0 = nn.Sequential(nn.Conv2d(f0, f1, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb1 = GroupedConvBlock(f1)
        self.down1 = nn.Sequential(nn.Conv2d(f1, f2, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb2 = GroupedConvBlock(f2)
        self.down2 = nn.Sequential(nn.Conv2d(f2, f3, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb3 = GroupedConvBlock(f3)
        self.down3 = nn.Sequential(nn.Conv2d(f3, f4, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.bottleneck = GroupedConvBlock(f4)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Conv2d(f4 + f3, f3, 3, padding=1)
        self.db3 = GroupedConvBlock(f3)
        self.up2 = nn.Conv2d(f3 + f2, f2, 3, padding=1)
        self.db2 = GroupedConvBlock(f2)
        self.up1 = nn.Conv2d(f2 + f1, f1, 3, padding=1)
        self.db1 = GroupedConvBlock(f1)
        self.up0 = nn.Conv2d(f1 + f0, f0, 7, padding=3)
        self.db0 = GroupedConvBlock(f0)
        self.out_conv = nn.Conv2d(f0, 3, 7, padding=3)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        e0 = self.eb0(self.in_conv(x))
        e1 = self.eb1(self.down0(e0))
        e2 = self.eb2(self.down1(e1))
        e3 = self.eb3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.db3(self.up3(torch.cat([self.upsample(b), e3], 1)))
        d2 = self.db2(self.up2(torch.cat([self.upsample(d3), e2], 1)))
        d1 = self.db1(self.up1(torch.cat([self.upsample(d2), e1], 1)))
        d0 = self.db0(self.up0(torch.cat([self.upsample(d1), e0], 1)))
        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
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
