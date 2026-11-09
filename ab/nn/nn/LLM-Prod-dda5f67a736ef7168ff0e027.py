import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ELUAvgBlock(nn.Module):
    """ELU block with an AvgPool2d context branch."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ctx_pool = nn.AvgPool2d(4)
        self.ctx = nn.Conv2d(channels, channels, 1)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        c = self.ctx(self.ctx_pool(out))
        c = nn.functional.interpolate(c, size=out.shape[-2:], mode="nearest")
        return self.conv2(out + c) + x

class LPPoolBlock(nn.Module):
    """LPPool2d context branch + Hardsigmoid gate."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.pool = nn.LPPool2d(2, 4)
        self.ctx = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Hardsigmoid(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        h = self.conv1(x)
        c = self.ctx(self.pool(h))
        c = nn.functional.interpolate(c, size=h.shape[-2:], mode="nearest")
        return self.conv2(h * self.gate(c)) + x



def supported_hyperparameters():
    return {"lr"}

class HaarDown(nn.Module):
    def __init__(self, c):
        super().__init__()
        base = torch.tensor([[[1., 1.], [1., 1.]],
                             [[1., -1.], [1., -1.]],
                             [[1., 1.], [-1., -1.]],
                             [[1., -1.], [-1., 1.]]], dtype=torch.float32) / 2.0
        self.register_buffer("w", base.unsqueeze(1).repeat(c, 1, 1, 1))
        self.c = c

    def forward(self, x):
        return F.conv2d(x, self.w, stride=2, groups=self.c)


class DilBlock(nn.Module):
    """Multi-dilation channel-preserving residual block (crossover contract)."""

    def __init__(self, channels):
        super().__init__()
        self.d1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.d4 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.d1(x))
        h = self.act(self.d2(h))
        return self.act(self.d4(h) + x)


class NLBlock(nn.Module):
    """Windowed non-local aggregation (N3Net / NLRN family): each pixel is replaced by a
    weighted average of its 7x7 neighbourhood, with weights computed from FEATURE SIMILARITY
    -- a learned, differentiable BM3D-style collaborative filter. The contrast with
    DenoiseKPN is where the weights come from: KPN predicts them with a conv head; here they
    ARE the query-key dot products, so self-similar structure (edges, texture) reinforces
    itself and incoherent noise cancels.

    Computed as 49 shifted views of a zero-padded tensor rather than one F.unfold: unfold
    materialises a (B,C,49,H,W) buffer (~5 GB at batch 16), which does not fit a 24 GB card;
    the shifted form peaks at the (B,49,H,W) similarity map instead. Same math, same output."""

    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        q = self.q(x)                                              # (B,C,H,W)
        kp = F.pad(self.k(x), (3, 3, 3, 3))
        vp = F.pad(self.v(x), (3, 3, 3, 3))
        logits = []
        for dy in range(7):
            for dx in range(7):
                logits.append((q * kp[:, :, dy:dy + h, dx:dx + w]).sum(1) * self.scale)
        sim = torch.softmax(torch.stack(logits, dim=1), dim=1)     # (B,49,H,W)
        out = x
        i = 0
        for dy in range(7):
            for dx in range(7):
                out = out + vp[:, :, dy:dy + h, dx:dx + w] * sim[:, i].unsqueeze(1)
                i += 1
        return out


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
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        self.h1 = HaarDown(ch)
        self.h2 = HaarDown(4 * ch)
        f = 16 * ch
        self.enc = nn.Conv2d(f, f, 3, padding=1)
        self.body = nn.Sequential(DilBlock(f), DilBlock(f), DilBlock(f), ELUAvgBlock(f), LPPoolBlock(f), NLBlock(f), MDTA(f, 2))
        self.dec = nn.Conv2d(f, f, 3, padding=1)
        self.up1 = nn.PixelShuffle(2)
        self.up2 = nn.PixelShuffle(2)
        self.s1 = nn.Conv2d(ch, 16, 3, padding=1)
        self.s2 = nn.Conv2d(16, 16, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.r1 = nn.Conv2d(16 + ch, 32, 3, padding=1)
        self.r2 = nn.Conv2d(32, ch, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        t = self.enc(self.h2(self.h1(x)))
        t = self.body(t)
        t = self.dec(self.body(t))
        t = self.up2(self.up1(t))
        s = self.act(self.s2(self.act(self.s1(x))))
        r = self.act(self.r1(torch.cat([s, t], dim=1)))
        return torch.clamp(self.r2(r) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        bn_layers = [m for m in self.modules() if isinstance(
            m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))]
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            bn_state = [(m.running_mean.clone(), m.running_var.clone(),
                         m.num_batches_tracked.clone()) for m in bn_layers]
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            bad = not torch.isfinite(loss)
            if not bad and bn_layers:
                bad = not all(torch.isfinite(m.running_mean).all()
                              and torch.isfinite(m.running_var).all() for m in bn_layers)
            if bad:
                for m, (rm, rv, nb) in zip(bn_layers, bn_state):
                    m.running_mean.copy_(rm)
                    m.running_var.copy_(rv)
                    m.num_batches_tracked.copy_(nb)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
