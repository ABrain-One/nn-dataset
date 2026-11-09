import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}

class LiteDenoisingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.conv2 = nn.Conv2d(mid, out_channels, 5, padding=2)
        self.actv = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.actv(self.conv1(x))
        out = self.conv2(out)
        return self.actv(out + identity)

class MDTA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
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

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        channels = in_shape[1]
        dim = 32
        heads = 2

        self.intro = nn.Conv2d(channels, dim, 3, padding=1)
        self.enc0 = LiteDenoisingBlock(dim, dim)
        self.enc1 = LiteDenoisingBlock(dim, dim)
        self.enc2 = LiteDenoisingBlock(dim, dim)
        self.enc3 = LiteDenoisingBlock(dim, dim)
        self.down0 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.down1 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.latent = MDTA(dim, heads)
        self.up2 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.up0 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.dec0 = LiteDenoisingBlock(dim, dim)
        self.dec1 = LiteDenoisingBlock(dim, dim)
        self.dec2 = LiteDenoisingBlock(dim, dim)
        self.ending = nn.Conv2d(dim, channels, 3, padding=1)

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
        s2 = self.enc2(y)
        y = self.down2(s2)
        y = self.latent(y)
        y = self.up2(y)
        y = self.dec0(self.up0(y))
        y = self.up1(y)
        y = self.dec1(self.up0(s2))
        y = self.up0(y)
        y = self.dec2(self.up0(s1))
        y = self.ending(y)
        y = y[:, :, :h, :w]
        return torch.clamp(y + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)

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
