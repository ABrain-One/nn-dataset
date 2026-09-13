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
        self.conv2 = nn.Conv2d(mid, out_channels, 3, padding=1)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.actv(self.conv1(x))
        out = self.conv2(out)
        return self.actv(out + identity)

class GroupNormLayer(nn.Module):
    def __init__(self, channels, num_groups=16, eps=1e-5):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=eps)

    def forward(self, x):
        return self.group_norm(x)

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

class GDFN(nn.Module):
    def __init__(self, dim, expand=2.0):
        super().__init__()
        hidden = int(dim * expand)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.project_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        x = self.dw(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = GroupNormLayer(dim)
        self.attn = MDTA(dim, heads)
        self.norm2 = GroupNormLayer(dim)
        self.ffn = GDFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

def _down(c):
    return nn.Sequential(nn.Conv2d(c, c // 2, 3, 1, 1, bias=False), nn.PixelUnshuffle(2))

def _up(c):
    return nn.Sequential(nn.Conv2d(c, c * 2, 3, 1, 1, bias=False), nn.PixelShuffle(2))

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        ch = in_shape[1] if len(in_shape) >= 3 else 3
        dim = 32
        heads = 2

        self.intro = nn.Conv2d(ch, dim, 3, padding=1)
        self.enc0 = TransformerBlock(dim, heads)
        self.down0 = _down(dim)
        self.enc1 = LiteDenoisingBlock(2 * dim, 2 * dim)
        self.down1 = _down(2 * dim)
        self.latent = TransformerBlock(4 * dim, heads)
        self.up1 = _up(4 * dim)
        self.reduce1 = nn.Conv2d(4 * dim, 2 * dim, 1)
        self.dec1 = LiteDenoisingBlock(2 * dim, 2 * dim)
        self.up0 = _up(2 * dim)
        self.reduce0 = nn.Conv2d(2 * dim, dim, 1)
        self.dec0 = TransformerBlock(dim, heads)
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
