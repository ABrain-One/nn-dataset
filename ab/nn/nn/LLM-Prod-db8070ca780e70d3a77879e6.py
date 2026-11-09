import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FourierUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Conv2d(channels * 2, channels * 2, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        spec = torch.fft.rfft2(x, norm="ortho")
        z = torch.view_as_real(spec)
        b, c = z.shape[0], z.shape[1]
        fh, fw = z.shape[2], z.shape[3]
        z = z.permute(0, 1, 4, 2, 3).reshape(b, c * 2, fh, fw)
        z = self.act(self.mix(z))
        z = z.reshape(b, c, 2, fh, fw).permute(0, 1, 3, 4, 2).contiguous()
        out = torch.fft.irfft2(torch.view_as_complex(z), s=(h, w), norm="ortho")
        return out + x


class ELUAvgBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 1, padding=0)
        self.ctx_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.ctx = nn.Conv2d(channels, channels, 5, padding=2)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        c = self.ctx(self.ctx_pool(out))
        c = F.interpolate(c, size=out.shape[-2:], mode='nearest')
        return self.conv2(out + c) + x





class DualPathBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.conv_ctx = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        h = self.act(self.conv1(x))
        ctx = self.avg_pool(x)
        ctx = self.conv_ctx(ctx)
        ctx = F.interpolate(ctx, size=h.shape[-2:], mode='nearest')
        h = h + ctx * self.gate(ctx)
        return self.act(self.conv2(h) + x)


def _down(c):
    return nn.Sequential(nn.Conv2d(c, c // 2, 3, 1, 1, bias=False), nn.PixelUnshuffle(2))


def _up(c):
    return nn.Sequential(nn.Conv2d(c, c * 2, 3, 1, 1, bias=False), nn.PixelShuffle(2))


def supported_hyperparameters():
    return {'lr'}


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        ch = in_shape[1] if len(in_shape) >= 3 else 3
        dim = 32
        heads = 2

        self.intro = nn.Conv2d(ch, dim, 3, padding=1)
        self.enc0 = DualPathBlock(dim)
        self.down0 = _down(dim)
        self.enc1 = ELUAvgBlock(2 * dim)
        self.down1 = _down(2 * dim)
        self.latent = FourierUnit(4 * dim)
        self.up1 = _up(4 * dim)
        self.reduce1 = nn.Conv2d(4 * dim, 2 * dim, 1)
        self.dec1 = DualPathBlock(2 * dim)
        self.up0 = _up(2 * dim)
        self.reduce0 = nn.Conv2d(2 * dim, dim, 1)
        self.dec0 = ELUAvgBlock(dim)
        self.ending = nn.Conv2d(dim, ch, 3, padding=1)

        self.padder = 4
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def _pad(self, x):
        _, _, h, w = x.shape
        p = self.padder
        ph = (p - h % p) % p
        pw = (p - w % p) % p
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        return x, h, w

    def forward(self, x):
        identity = x
        y, h, w = self._pad(x)
        y = self.intro(y)
        s0 = self.enc0(y)
        y = self.down0(s0)
        s1 = self.enc1(y)
        y = self.down1(s1)
        y = self.latent(y)
        y = self.up1(y)
        y = self.dec1(self.reduce1(torch.cat([y, s1], 1)))
        y = self.up0(y)
        y = self.dec0(self.reduce0(torch.cat([y, s0], 1)))
        y = self.ending(y)
        y = y[:, :, :h, :w]
        return torch.clamp(y + identity, 0.0, 1.0)

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
