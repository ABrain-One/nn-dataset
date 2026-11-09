import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DualPathBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(2)
        self.conv_ctx = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        h = self.act(self.conv1(x))
        ctx = self.avg_pool(x)
        ctx = self.conv_ctx(ctx)
        ctx = F.interpolate(ctx, size=h.shape[-2:], mode='nearest')
        h = h + ctx * self.gate(ctx)
        return self.act(self.conv2(h) + x)

class FourierUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.basis = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(4)])
        self.gate = nn.Conv2d(channels, 4, 3, padding=1)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        g = torch.softmax(self.gate(x), dim=1)
        out = 0.0
        for i, conv in enumerate(self.basis):
            out = out + conv(x) * g[:, i:i + 1]
        return x + self.act(out + x)

class MBConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = channels * 2
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.project = nn.Conv2d(hidden, channels, 1)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.act(self.expand(x))
        out = self.act(self.dw(out))
        return x + self.project(out)

def supported_hyperparameters():
    return {'lr'}

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        dim = 32
        heads = 2

        self.intro = nn.Conv2d(ch, dim, 3, padding=1)
        self.enc0 = FourierUnit(dim)
        self.down0 = nn.Conv2d(dim, dim // 2, 3, stride=2, padding=1)
        self.enc1 = DualPathBlock(dim // 2)
        self.down1 = nn.Conv2d(dim // 2, dim // 4, 3, stride=2, padding=1)
        self.latent = MBConvBlock(dim // 4)
        self.up1 = nn.ConvTranspose2d(dim // 4, dim // 2, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(dim // 2, dim, 3, stride=2, padding=1, output_padding=1)
        self.ending = nn.Conv2d(dim, ch, 3, padding=1)

        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def _pad(self, x):
        _, _, h, w = x.shape
        p = 4
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
        y = self.up2(y)
        y = self.ending(y)
        y = y[:, :, :h, :w]
        return torch.clamp(y + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 0.0001)
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
