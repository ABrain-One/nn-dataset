import torch
import torch.nn as nn
import torch.optim as optim

class LPPoolBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        c = channels
        self.norm1 = Xb__Norm2d(c)
        self.proj_in = nn.Conv2d(c, c * 2, 1)
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.proj_out = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1) * 0.01)
        self.norm2 = Xb__Norm2d(c)
        self.ffn_up = nn.Conv2d(c, c * 4, 1)
        self.ffn_dn = nn.Conv2d(c * 2, c, 1)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 0.01)

    def forward(self, inp):
        x = self.norm1(inp)
        xp = self.proj_in(x)
        x1, x2 = xp.chunk(2, dim=1)
        x1 = self.dw(x1)
        x = x1 * x2
        x = x * self.sca(x)
        x = self.proj_out(x)
        y = inp + x * self.beta
        x = self.norm2(y)
        x = self.ffn_up(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.ffn_dn(x)
        return y + x * self.gamma

class Xb__Norm2d(nn.Module):
    """Mobile-GPU-clean norm: BatchNorm2d folds into the preceding conv on the
    TFLite GPU delegate. LayerNorm (the NAFNet default) + the permute round-trip
    are only partially supported and force CPU fallbacks."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + self.norm(x)

class Net(nn.Module):
    """LPPool — introduces op types absent from the pool; crossover-capable channel-preserving block."""

    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        Block = LPPoolBlock
        self.in_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.eb0 = Block(16)
        self.unshuf0 = nn.PixelUnshuffle(2)
        self.down0 = nn.Conv2d(64, 32, 3, padding=1)
        self.eb1 = Block(32)
        self.unshuf1 = nn.PixelUnshuffle(2)
        self.down1 = nn.Conv2d(128, 64, 3, padding=1)
        self.eb2 = Block(64)
        self.unshuf2 = nn.PixelUnshuffle(2)
        self.down2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bottleneck = Block(128)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.Conv2d(192, 64, 3, padding=1)
        self.db2 = Block(64)
        self.up1 = nn.Conv2d(96, 32, 3, padding=1)
        self.db1 = Block(32)
        self.up0 = nn.Conv2d(48, 16, 3, padding=1)
        self.db0 = Block(16)
        self.out_conv = nn.Conv2d(16, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        e0 = self.in_conv(x)
        e0 = self.eb0(e0)
        e1 = self.down0(self.unshuf0(e0))
        e1 = self.eb1(e1)
        e2 = self.down1(self.unshuf1(e1))
        e2 = self.eb2(e2)
        bt = self.down2(self.unshuf2(e2))
        bt = self.bottleneck(bt)
        d2 = self.db2(self.up2(torch.cat([self.upsample(bt), e2], 1)))
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
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)