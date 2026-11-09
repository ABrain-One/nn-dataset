import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}

class DilBlock(nn.Module):
    """Multi-dilation channel-preserving residual block (crossover contract)."""
    def __init__(self, channels):
        super().__init__()
        self.d1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.d4 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.d1(x))
        h = self.act(self.d2(h))
        return self.act(self.d4(h) + x)

class EluAvgBlock(nn.Module):
    """LPPool2d context branch + Hardsigmoid gate."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, padding=0)
        self.pool = nn.LPPool2d(2, 4)
        self.ctx = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Hardsigmoid(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 5, padding=2)

    def forward(self, x):
        h = self.conv1(x)
        c = self.ctx(self.pool(h))
        c = nn.functional.interpolate(c, size=h.shape[-2:], mode='nearest')
        return self.conv2(h * self.gate(c)) + x

class AxialBlock(nn.Module):
    """Fast-Fourier-Convolution unit (FFC / LaMa / DeepRFT family): transform to the
    frequency domain, apply a learned pointwise mix to the spectrum, transform back. One
    frequency-domain multiply touches EVERY pixel, so the receptive field is global in a
    single op rather than grown layer by layer."""
    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Conv2d(channels * 2, channels * 2, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h, w = (x.shape[2], x.shape[3])
        spec = torch.fft.rfft2(x, norm='ortho')
        z = torch.view_as_real(spec)
        b, c = (z.shape[0], z.shape[1])
        fh, fw = (z.shape[2], z.shape[3])
        z = z.permute(0, 1, 4, 2, 3).reshape(b, c * 2, fh, fw)
        z = self.act(self.mix(z))
        z = z.reshape(b, c, 2, fh, fw).permute(0, 1, 3, 4, 2).contiguous()
        out = torch.fft.irfft2(torch.view_as_complex(z), s=(h, w), norm='ortho')
        return out + x

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, f, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, f, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ax1 = AxialBlock(f)
        self.ax2 = AxialBlock(f)
        self.dil1 = DilBlock(f)
        self.dil2 = DilBlock(f)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(f, 24, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.local = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.elu1 = EluAvgBlock(24)
        self.elu2 = EluAvgBlock(24)
        self.tail = nn.Conv2d(48, 3, 5, padding=2)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        h = self.stem(x)
        h = self.ax1(h)
        h = self.dil1(h)
        h = self.ax2(h)
        h = self.dil2(h)
        g = self.up(h)
        l = self.local(x)
        g = self.elu1(g)
        l = self.elu2(l)
        return torch.clamp(self.tail(torch.cat([g, l], dim=1)) + identity, 0.0, 1.0)

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
