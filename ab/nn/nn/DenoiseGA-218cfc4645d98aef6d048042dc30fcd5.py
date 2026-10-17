# name=DenoiseMFDNet
# Efficient SOTA seed: MFDNet family ("Lightweight network towards real-time image denoising
# on mobile devices", arXiv 2211.04687). Faithful port of the DEFINING mechanism (not
# weight-exact): fixed HAAR wavelet downsampling (invertible, lossless, cheap) to work at low
# resolution, a stack of Mobile-Friendly Denoise Blocks = K Conv3x3+LReLU + a RepConv
# (reparameterizable 1x1-3x3-1x1 + skip) + a Mobile-Friendly Attention (downsample->3x3->ReLU->
# bilinear-up->gate), then PixelShuffle up. Global residual + clamp. Trace-clean (Haar is a
# fixed grouped conv; no data-dependent control flow) and edge-legal.
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class HaarDown(nn.Module):
    """One-level 2x Haar DWT as a fixed (non-learnable) grouped conv: C -> 4C at half res."""

    def __init__(self, c):
        super().__init__()
        base = torch.tensor([[[1., 1.], [1., 1.]],      # LL
                             [[1., -1.], [1., -1.]],     # HL
                             [[1., 1.], [-1., -1.]],     # LH
                             [[1., -1.], [-1., 1.]]], dtype=torch.float32) / 2.0
        w = base.unsqueeze(1).repeat(c, 1, 1, 1)         # (4C,1,2,2), groups=C
        self.register_buffer("w", w)
        self.c = c

    def forward(self, x):
        return Fn.conv2d(x, self.w, stride=2, groups=self.c)


class RepConv(nn.Module):
    """Training-time reparameterizable block: 1x1 -> 3x3 -> 1x1 + skip (merges to one 3x3)."""

    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 1)
        self.c3 = nn.Conv2d(c, c, 3, padding=1)
        self.c1b = nn.Conv2d(c, c, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.c1b(self.c3(self.c1(x))) + x)


class MFA(nn.Module):
    """Mobile-Friendly Attention: downsample -> 3x3 -> ReLU -> bilinear up -> sigmoid gate."""

    def __init__(self, c):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        a = self.act(self.conv(self.pool(x)))
        a = Fn.interpolate(a, scale_factor=2, mode="bilinear", align_corners=False)
        return x * torch.sigmoid(a)


class MFDB(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.body = nn.Sequential(*[nn.Sequential(nn.Conv2d(c, c, 3, padding=1),
                                                  nn.LeakyReLU(0.2, inplace=True)) for _ in range(k)])
        self.rep = RepConv(c)
        self.mfa = MFA(c)

    def forward(self, x):
        return self.mfa(self.rep(self.body(x))) + x


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=(3, 256, 256), prm={}, device="cuda"):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        self.h1 = HaarDown(ch)                 # ch -> 4*ch, /2
        self.h2 = HaarDown(4 * ch)             # 4ch -> 16*ch, /4
        f = 16 * ch                            # body width (48 for RGB)
        self.head = nn.Conv2d(f, f, 3, padding=1)
        self.blocks = nn.Sequential(*[MFDB(f) for _ in range(3)])   # M=3
        self.tail = nn.Conv2d(f, f, 3, padding=1)
        self.up1 = nn.PixelShuffle(2)          # f -> f/4, x2   (48 -> 12, /2)
        self.up2 = nn.PixelShuffle(2)          # f/4 -> f/16, x2 (12 -> 3, full)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        h = self.h2(self.h1(x))                 # -> 16ch, /4
        h = self.head(h)
        h = self.blocks(h)
        h = self.tail(h)
        h = self.up2(self.up1(h))               # -> ch, full res
        return torch.clamp(h + identity, 0.0, 1.0)

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
                    m.running_mean.copy_(rm); m.running_var.copy_(rv); m.num_batches_tracked.copy_(nb)
                continue
            if not torch.isfinite(loss): continue  # NaN-guard (yield>90%)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
