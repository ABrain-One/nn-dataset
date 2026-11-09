import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {'lr'}


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LinearMixBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.pool = nn.MaxPool2d(4)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        qh = self.pool(self.q(x)).flatten(2)
        kh = self.pool(self.k(x)).flatten(2)
        vh = self.pool(self.v(x)).flatten(2)
        att = self.soft(torch.matmul(qh.transpose(1, 2), kh) / c ** 0.5)
        out = torch.matmul(vh, att.transpose(1, 2))
        out = out.reshape(b, c, h // 4, w // 4)
        out = F.interpolate(out, size=(h, w), mode='nearest')
        return x + self.proj(out)


def _down(c):
    return nn.Sequential(nn.Conv2d(c, c // 2, 3, 1, 1, bias=False), nn.PixelUnshuffle(2))


def _up(c):
    return nn.Sequential(nn.Conv2d(c, c * 2, 3, 1, 1, bias=False), nn.PixelShuffle(2))


class DenoiserBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.se_block = SEBlock(channels)
        self.act = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.act(self.conv1(x))
        out = self.se_block(out)
        out = self.act(self.conv2(out))
        return out + identity


class GatedAdaptiveBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c = channels

        self.norm1 = LayerNorm2d(c)
        self.proj_in = nn.Conv2d(c, c * 2, 1)
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.proj_out = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1) * 0.01)

        self.norm2 = LayerNorm2d(c)
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


class KMixBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.basis = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=1) for _ in range(4)])
        self.gate = nn.Conv2d(channels, 4, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        g = torch.softmax(self.gate(x), dim=1)
        out = 0.0
        for i, conv in enumerate(self.basis):
            out = out + conv(x) * g[:, i:i + 1]
        return self.act(out + x)


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        ch = in_shape[1] if len(in_shape) >= 3 else 3
        dim = 32

        self.intro = nn.Conv2d(ch, dim, 3, padding=1)
        self.enc0 = LinearMixBlock(dim)
        self.down0 = _down(dim)
        self.enc1 = DenoiserBlock(2 * dim)
        self.down1 = _down(2 * dim)
        self.latent = GatedAdaptiveBlock(4 * dim)
        self.up1 = _up(4 * dim)
        self.reduce1 = nn.Conv2d(4 * dim, 2 * dim, 1)
        self.dec1 = DenoiserBlock(2 * dim)
        self.up0 = _up(2 * dim)
        self.reduce0 = nn.Conv2d(2 * dim, dim, 1)
        self.dec0 = DenoiserBlock(dim)
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
