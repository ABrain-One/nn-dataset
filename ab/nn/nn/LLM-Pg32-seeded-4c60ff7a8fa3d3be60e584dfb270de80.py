import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}

class ELUAddSkip(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        if down:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.ELU(inplace=True)

    def forward(self, x, skip=None):
        y = self.act(self.conv1(x))
        if skip is not None:
            y = self.skip_conv(skip) + y
        y = self.conv2(y)
        return y

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
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = MDTA(dim, heads)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = GDFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        dim = 32
        heads = 2

        self.intro = nn.Conv2d(ch, dim, 3, padding=1)
        self.enc0 = ELUAddSkip(dim, dim)
        self.down0 = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.enc1 = ELUAddSkip(dim, dim)
        self.down1 = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.latent = TransformerBlock(dim, heads)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(dim, dim, 2, stride=2), nn.ReLU(inplace=True))
        self.dec1 = ELUAddSkip(dim, dim)
        self.up0 = nn.Sequential(nn.ConvTranspose2d(dim, dim, 2, stride=2), nn.ReLU(inplace=True))
        self.dec0 = ELUAddSkip(dim, dim)
        self.ending = nn.Conv2d(dim, ch, 3, padding=1)

        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        y = self.intro(x)
        s0 = self.enc0(y)
        y = self.down0(s0)
        s1 = self.enc1(y)
        y = self.down1(s1)
        y = self.latent(y)
        y = self.up1(y)
        y = self.dec1(y)
        y = self.up0(y)
        y = self.dec0(y)
        y = self.ending(y)
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
