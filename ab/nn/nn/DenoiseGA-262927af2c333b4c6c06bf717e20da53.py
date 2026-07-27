# promoted name=DenoiseLapPyrXDenoiseLPPool psnr50=34.310 macs=915996672
# name=DenoiseLapPyrXDenoiseLPPool parent=DenoiseLapPyrxDenoiseLPPool checksum=dad0218709203ada
# psnr_db=32.0196 eff_db=32.4430 macs=915996672 params=40578
import torch
import torch.nn as nn
import torch.optim as optim

class LBlock(nn.Module):
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
        c = nn.functional.interpolate(c, size=h.shape[-2:], mode='nearest')
        return self.conv2(h * self.gate(c)) + x

class Net(nn.Module):
    """LAPLACIAN-PYRAMID denoiser. The input is split into a low-frequency band (AvgPool)
    and a detail band (input minus upsampled low band); each band is denoised by its OWN
    small network IN PARALLEL, then the bands are recomposed. Not an encoder->decoder and
    not a flat stack: the graph is parallel per-frequency-band branches."""

    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        fh, fl = (12, 24)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.lo_in = nn.Conv2d(3, fl, 3, padding=1)
        self.lo_b1 = LBlock(fl)
        self.lo_b2 = LBlock(fl)
        self.lo_b3 = LBlock(fl)
        self.lo_out = nn.Conv2d(fl, 3, 3, padding=1)
        self.hi_in = nn.Conv2d(3, fh, 3, padding=1)
        self.hi_b1 = LBlock(fh)
        self.hi_b2 = LBlock(fh)
        self.hi_out = nn.Conv2d(fh, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        lo = self.pool(x)
        band = x - self.up(lo)
        dlo = lo + self.lo_out(self.lo_b3(self.lo_b2(self.lo_b1(self.lo_in(lo)))))
        dband = self.hi_out(self.hi_b2(self.hi_b1(self.hi_in(band))))
        return torch.clamp(self.up(dlo) + dband, 0.0, 1.0)

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
            if not torch.isfinite(loss): continue  # NaN-guard (yield>90%)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)