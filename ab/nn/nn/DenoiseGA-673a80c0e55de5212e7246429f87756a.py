import torch
import torch.nn as nn
import torch.optim as optim

class LBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        c = channels
        self.norm1 = Xb_Xb_Xb__Norm2d(c)
        self.proj_in = nn.Conv2d(c, c * 2, 1)
        self.dw = nn.Conv2d(c, c, 5, padding=2, groups=c)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.proj_out = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1) * 0.01)
        self.norm2 = Xb_Xb_Xb__Norm2d(c)
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
        return inp + (y + x * self.gamma)

class Xb_Xb_Xb__Norm2d(nn.Module):
    """Mobile-GPU-clean norm: BatchNorm2d folds into the preceding conv on the
    TFLite GPU delegate. LayerNorm (the NAFNet default) + the permute round-trip
    are only partially supported and force CPU fallbacks."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + self.norm(x)

class Net(nn.Module):
    """LAPLACIAN-PYRAMID denoiser. The input is split into a low-frequency band (AvgPool)
    and a detail band (input minus upsampled low band); each band is denoised by its OWN
    small network IN PARALLEL, then the bands are recomposed. Not an encoder->decoder and
    not a flat stack: the graph is parallel per-frequency-band branches."""

    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        fh, fl = (12, 24)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.lo_in = nn.Conv2d(3, fl, 3, padding=1)
        self.lo_b1 = LBlock(fl)
        self.lo_b2 = LBlock(fl)
        self.lo_b3 = LBlock(fl)
        self.lo_out = nn.Conv2d(fl, 3, 3, padding=1)
        self.hi_in = nn.Conv2d(3, fh, 3, padding=1)
        self.hi_b1 = LBlock(fh)
        self.hi_b2 = LBlock(fh)
        self.hi_out = nn.Conv2d(fh, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        lo = self.pool(x)
        band = x - self.up(lo)
        dlo = lo + self.lo_out(self.lo_b3(self.lo_b2(self.lo_b1(self.lo_in(lo)))))
        dband = self.hi_out(self.hi_b2(self.hi_b1(self.hi_in(band))))
        return torch.clamp(self.up(dlo) + dband, 0.0, 1.0)

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