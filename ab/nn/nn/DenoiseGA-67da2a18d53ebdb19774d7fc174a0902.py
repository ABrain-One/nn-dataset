import torch
import torch.nn as nn
import torch.optim as optim

class SPBlock(nn.Module):
    """Self-attention residual block (softmax + matmul — absent from the pool)."""

    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 5, padding=2)
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
        out = nn.functional.interpolate(out, size=(h, w), mode='nearest')
        return x + self.proj(out)

class Net(nn.Module):
    """SUB-PIXEL (FFDNet-style) denoiser — NOT a U-Net. The only resolution change is a
    single PixelUnshuffle at the input and PixelShuffle at the output; the body is a FLAT
    stack of residual blocks at reduced resolution. No encoder/decoder, no cross-scale skips."""

    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 48
        self.unshuffle = nn.PixelUnshuffle(2)
        self.in_conv = nn.Conv2d(3 * 4, f, 1, padding=0)
        self.b0 = SPBlock(f)
        self.b1 = SPBlock(f)
        self.b2 = SPBlock(f)
        self.b3 = SPBlock(f)
        self.b4 = SPBlock(f)
        self.b5 = SPBlock(f)
        self.out_conv = nn.Conv2d(f, 3 * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        h = self.in_conv(self.unshuffle(x))
        h = self.b0(h)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        h = self.b4(h)
        h = self.b5(h)
        return torch.clamp(self.shuffle(self.out_conv(h)) + identity, 0.0, 1.0)

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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)