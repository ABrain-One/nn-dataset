import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}

class HaarDown(nn.Module):
    """One-level 2x Haar DWT as a fixed grouped conv: C -> 4C at half res."""

    def __init__(self, c):
        super().__init__()
        base = torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[1.0, -1.0], [1.0, -1.0]], [[1.0, 1.0], [-1.0, -1.0]], [[1.0, -1.0], [-1.0, 1.0]]], dtype=torch.float32) / 2.0
        self.register_buffer('w', base.unsqueeze(1).repeat(c, 1, 1, 1))
        self.c = c

    def forward(self, x):
        return Fn.conv2d(x, self.w, stride=2, groups=self.c)

class ConvBlock(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, 15, 1, padding=0)
        self.c2 = nn.Conv2d(15, c, 3, padding=1)
        self.act = nn.ELU(0.2, inplace=True)

    def forward(self, x):
        return x + self.act(self.c2(self.act(self.c1(x))) + x)

class Net(nn.Module):

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=(3, 256, 256), prm={}, device='cuda'):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        self.h1 = HaarDown(ch)
        self.h2 = HaarDown(4 * ch)
        f = 16 * ch
        self.enc = nn.Conv2d(f, f, 3, padding=1)
        self.body = nn.Sequential(ConvBlock(f), ConvBlock(f), ConvBlock(f))
        self.dec = nn.Conv2d(f, f, 3, padding=1)
        self.up1 = nn.PixelShuffle(2)
        self.up2 = nn.PixelShuffle(2)
        self.s1 = nn.Conv2d(ch, 16, 3, padding=1)
        self.s2 = nn.Conv2d(16, 16, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.r1 = nn.Conv2d(16 + ch, 32, 3, padding=1)
        self.r2 = nn.Conv2d(32, ch, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        t = self.enc(self.h2(self.h1(x)))
        t = self.dec(self.body(t))
        t = self.up2(self.up1(t))
        s = self.act(self.s2(self.act(self.s1(x))))
        r = self.act(self.r1(torch.cat([s, t], dim=1)))
        return torch.clamp(self.r2(r) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        bn_layers = [m for m in self.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))]
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
            self.optimizer.zero_grad()
            bn_state = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone()) for m in bn_layers]
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            bad = not torch.isfinite(loss)
            if not bad and bn_layers:
                bad = not all((torch.isfinite(m.running_mean).all() and torch.isfinite(m.running_var).all() for m in bn_layers))
            if bad:
                for m, (rm, rv, nb) in zip(bn_layers, bn_state):
                    m.running_mean.copy_(rm)
                    m.running_var.copy_(rv)
                    m.num_batches_tracked.copy_(nb)
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