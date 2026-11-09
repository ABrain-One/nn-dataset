import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}



class AttentionBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        hd = c // self.heads
        q = q.reshape(b, self.heads, hd, h * w)
        k = k.reshape(b, self.heads, hd, h * w)
        v = v.reshape(b, self.heads, hd, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, c, h, w)
        return self.proj(out)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.9)

    def forward(self, x):
        net = self.act1(self.conv1(x))
        net = self.conv2(net)
        middle = x * self.alpha + net
        return middle + net

def _down(c):
    return nn.Sequential(nn.Conv2d(c, c // 2, 3, 1, 1, bias=False), nn.PixelUnshuffle(2))

def _up(c):
    return nn.Sequential(nn.Conv2d(c, c * 2, 3, 1, 1, bias=False), nn.PixelShuffle(2))

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        dim = 32
        heads = 2

        self.intro = nn.Conv2d(ch, dim, 3, padding=1)
        self.enc0 = AttentionBlock(dim, heads)
        self.down0 = _down(dim)
        self.enc1 = ResidualBlock(2 * dim)
        self.channel_attn1 = nn.Conv2d(2 * dim, 2 * dim, 1)
        self.down1 = _down(2 * dim)
        self.latent = AttentionBlock(4 * dim, heads)
        self.up1 = _up(4 * dim)
        self.reduce1 = nn.Conv2d(4 * dim, 2 * dim, 1)
        self.dec1 = ResidualBlock(2 * dim)
        self.channel_attn2 = nn.Conv2d(2 * dim, 2 * dim, 1)
        self.up0 = _up(2 * dim)
        self.reduce0 = nn.Conv2d(2 * dim, dim, 1)
        self.dec0 = AttentionBlock(dim, heads)
        self.ending = nn.Conv2d(dim, ch, 3, padding=1)

        self.criterion_mse = nn.MSELoss()
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
        s1 = self.channel_attn1(s1)
        y = self.down1(s1)
        y = self.latent(y)
        y = self.up1(y)
        y = self.dec1(self.reduce1(torch.cat([y, s1], 1)))
        y = self.channel_attn2(y)
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
            loss = loss_gt * 1000 + self.criterion_mse(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
