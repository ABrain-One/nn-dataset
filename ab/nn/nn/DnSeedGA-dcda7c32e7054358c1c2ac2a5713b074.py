import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class LBlock(nn.Module):
    """Channel-preserving residual block (crossover contract)."""

    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.c2(self.act(self.c1(x))) + x)


class Net(nn.Module):
    """LAPLACIAN-PYRAMID denoiser. The input is split into a low-frequency band (AvgPool)
    and a detail band (input minus upsampled low band); each band is denoised by its OWN
    small network IN PARALLEL, then the bands are recomposed. Not an encoder->decoder and
    not a flat stack: the graph is parallel per-frequency-band branches."""

    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        fh, fl = 12, 24
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
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
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5
        )

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
                # BN buffers were updated by this batch's forward BEFORE the
                # check — restore them or eval is poisoned while train stays ok
                for m, (rm, rv, nb) in zip(bn_layers, bn_state):
                    m.running_mean.copy_(rm)
                    m.running_var.copy_(rv)
                    m.num_batches_tracked.copy_(nb)
                continue
            if not torch.isfinite(loss): continue  # NaN-guard (yield>90%)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
