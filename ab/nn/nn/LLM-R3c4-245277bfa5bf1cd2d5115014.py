import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr'}

class DualChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1)).view(x.size(0), -1, 1, 1)
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1)).view(x.size(0), -1, 1, 1)
        out = avg_out + max_out
        return x * out.expand_as(x)

class DualLiteDenoisingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1)
        self.actv = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        skip = self.shortcut(x)
        h = self.actv(self.conv1(x))
        h = self.actv(self.conv2(h))
        h = self.conv3(h)
        return F.relu(h + skip)

class DualAxialBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Conv2d(channels * 2, channels * 2, 1)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        h, w = (x.shape[2], x.shape[3])
        spec = torch.fft.rfft2(x, norm='ortho')
        z = torch.view_as_real(spec)
        b, c = (z.shape[0], z.shape[1])
        fh, fw = (z.shape[2], z.shape[3])
        z = z.permute(0, 1, 4, 2, 3).reshape(b, c * 2, fh, fw)
        z = self.actv(self.mix(z))
        z = z.reshape(b, c, 2, fh, fw).permute(0, 1, 3, 4, 2).contiguous()
        out = torch.fft.irfft2(torch.view_as_complex(z), s=(h, w), norm='ortho')
        return out + x

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        channels = 3
        f0 = 16
        f1, f2, f3, f4 = (f0 * 2, f0 * 4, f0 * 8, f0 * 16)
        self.in_conv = nn.Conv2d(channels, f0, 5, padding=2)
        self.ldb0 = DualLiteDenoisingBlock(f0, f0)
        self.down0 = nn.Sequential(nn.Conv2d(f0, f1, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.ldb1 = DualLiteDenoisingBlock(f1, f1)
        self.down1 = nn.Sequential(nn.Conv2d(f1, f2, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.ldb2 = DualLiteDenoisingBlock(f2, f2)
        self.down2 = nn.Sequential(nn.Conv2d(f2, f3, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.ldb3 = DualLiteDenoisingBlock(f3, f3)
        self.down3 = nn.Sequential(nn.Conv2d(f3, f4, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.bottleneck = DualAxialBlock(f4)
        self.channel_att = DualChannelAttention(f4)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Conv2d(f4 + f3, f3, 3, padding=1)
        self.dlad3 = DualLiteDenoisingBlock(f3, f3)
        self.up2 = nn.Conv2d(f3 + f2, f2, 3, padding=1)
        self.dlad2 = DualLiteDenoisingBlock(f2, f2)
        self.up1 = nn.Conv2d(f2 + f1, f1, 3, padding=1)
        self.dlad1 = DualLiteDenoisingBlock(f1, f1)
        self.up0 = nn.Conv2d(f1 + f0, f0, 7, padding=3)
        self.dlad0 = DualLiteDenoisingBlock(f0, f0)
        self.out_conv = nn.Conv2d(f0, channels, 5, padding=2)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(self.device)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def forward(self, x):
        identity = x
        e0 = self.ldb0(self.in_conv(x))
        e1 = self.ldb1(self.down0(e0))
        e2 = self.ldb2(self.down1(e1))
        e3 = self.ldb3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        b = self.channel_att(b)
        d3 = self.dlad3(self.up3(torch.cat([self.upsample(b), e3], 1)))
        d2 = self.dlad2(self.up2(torch.cat([self.upsample(d3), e2], 1)))
        d1 = self.dlad1(self.up1(torch.cat([self.upsample(d2), e1], 1)))
        d0 = self.dlad0(self.up0(torch.cat([self.upsample(d1), e0], 1)))
        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

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