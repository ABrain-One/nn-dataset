import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SimpleChannelAtt(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return x * torch.sigmoid(self.fc(x.mean(dim=(2, 3), keepdim=True)))

class SpatialAtt(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = torch.amax(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


def supported_hyperparameters():
    return {"lr"}

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
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

def _down(c):
    return nn.Sequential(nn.Conv2d(c, c // 2, 3, 1, 1, bias=False), nn.PixelUnshuffle(2))

def _up(c):
    return nn.Sequential(nn.Conv2d(c, c * 2, 3, 1, 1, bias=False), nn.PixelShuffle(2))

class SplitNode(nn.Module):
    def __init__(self, f, level, depth):
        super().__init__()
        self.half = f // 2
        self.pad = nn.ReflectionPad2d(1)
        self.conv_a = nn.Conv2d(self.half, f, 3, stride=2)
        self.conv_b = nn.Conv2d(self.half, f, 3, stride=2)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.leaf = level >= depth - 1
        if self.leaf:
            self.mid_a = MidBlock(f)
            self.mid_b = MidBlock(f)
        else:
            self.child_a = SplitNode(f, level + 1, depth)
            self.child_b = SplitNode(f, level + 1, depth)
        self.up = nn.ConvTranspose2d(2 * f, f, 2, stride=2)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        skip = x
        a = self.act(self.conv_a(self.pad(x[:, :self.half])))
        b = self.act(self.conv_b(self.pad(x[:, self.half:])))
        if self.leaf:
            a = self.mid_a(a); b = self.mid_b(b)
        else:
            a = self.child_a(a); b = self.child_b(b)
        m = self.act2(self.up(torch.cat([a, b], dim=1)))
        return m + skip

class MidBlock(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(f, f, 3)
        self.conv2 = nn.Conv2d(f, f, 3)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.ca = SimpleChannelAtt(f)
        self.sa = SpatialAtt()

    def forward(self, x):
        n = self.ca(self.act(self.conv1(self.pad(x)))) + x
        return self.sa(self.act(self.conv2(self.pad(n)))) + n

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        ch = in_shape[1] if len(in_shape) >= 3 else 3
        dim = 32
        heads = 2

        self.intro = nn.Conv2d(ch, dim, 3, padding=1)
        self.enc0 = TransformerBlock(dim, heads)
        self.down0 = _down(dim)
        self.enc1 = TransformerBlock(2 * dim, heads)
        self.down1 = _down(2 * dim)
        self.latent = TransformerBlock(4 * dim, heads)
        self.up1 = _up(4 * dim)
        self.reduce1 = nn.Conv2d(4 * dim, 2 * dim, 1)
        self.dec1 = TransformerBlock(2 * dim, heads)
        self.up0 = _up(2 * dim)
        self.reduce0 = nn.Conv2d(2 * dim, dim, 1)
        self.dec0 = TransformerBlock(dim, heads)
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
        y = self.dec1(self.reduce1(torch.cat([y, s1], 1)))
        y = self.up0(y)
        y = self.dec0(self.reduce0(torch.cat([y, s0], 1)))
        y = self.ending(y)
        y = y[:, :, :h, :w]
        return torch.clamp(y + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 0.0001)
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
