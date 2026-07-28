import torch
import torch.nn as nn
import torch.optim as optim

class DilatedBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        c = channels
        self.norm1 = Xb_Xb_Xb_Xb__Norm2d(c)
        self.proj_in = nn.Conv2d(c, c * 2, 1)
        self.dw = nn.Conv2d(c, c, 5, padding=2, groups=c)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.proj_out = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1) * 0.01)
        self.norm2 = Xb_Xb_Xb_Xb__Norm2d(c)
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
        return inp + (y + x * self.gamma)

class Xb_Xb_Xb_Xb__Norm2d(nn.Module):
    """Mobile-GPU-clean norm: BatchNorm2d folds into the preceding conv on the
    TFLite GPU delegate. LayerNorm (the NAFNet default) + the permute round-trip
    are only partially supported and force CPU fallbacks."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + self.norm(x)

class Net(nn.Module):
    """PoolDeconv skeleton variant — crossover-capable channel-preserving block."""

    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        Block = DilatedBlock
        self.in_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.eb0_0 = Block(16)
        self.pool0 = nn.MaxPool2d(2)
        self.down0 = nn.Conv2d(16, 32, 3, padding=1)
        self.eb1_0 = Block(32)
        self.pool1 = nn.MaxPool2d(2)
        self.down1 = nn.Conv2d(32, 64, 3, padding=1)
        self.eb2_0 = Block(64)
        self.pool2 = nn.MaxPool2d(2)
        self.down2 = nn.Conv2d(64, 128, 7, padding=3)
        self.bottleneck = Block(128)
        self.deconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.up2 = nn.Conv2d(192, 64, 3, padding=1)
        self.db2_0 = Block(64)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.up1 = nn.Conv2d(96, 32, 3, padding=1)
        self.db1_0 = Block(32)
        self.deconv0 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.up0 = nn.Conv2d(48, 16, 3, padding=1)
        self.db0_0 = Block(16)
        self.out_conv = nn.Conv2d(16, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        e0 = self.in_conv(x)
        e0 = self.eb0_0(e0)
        e1 = self.down0(self.pool0(e0))
        e1 = self.eb1_0(e1)
        e2 = self.down1(self.pool1(e1))
        e2 = self.eb2_0(e2)
        bt = self.down2(self.pool2(e2))
        bt = self.bottleneck(bt)
        d2 = self.up2(torch.cat([self.deconv2(bt), e2], 1))
        d2 = self.db2_0(d2)
        d1 = self.up1(torch.cat([self.deconv1(d2), e1], 1))
        d1 = self.db1_0(d1)
        d0 = self.up0(torch.cat([self.deconv0(d1), e0], 1))
        d0 = self.db0_0(d0)
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
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)