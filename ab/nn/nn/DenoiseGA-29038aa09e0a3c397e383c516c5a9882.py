import torch
import torch.nn as nn
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}

class HaarDWT(nn.Module):
    """Fixed Haar wavelet transform. NOT learnable, NOT interpolation.

    Every other downsampler in the corpus either discards information (pool, stride) or is
    learned (strided conv). This is an orthogonal, invertible, FIXED basis: the four outputs
    are the LL, LH, HL and HH sub-bands, and IDWT reconstructs the input exactly. Noise lives
    predominantly in the high-frequency sub-bands, so separating them is a structural prior
    that no learned downsampler expresses.
    """

    def __init__(self):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x):
        p = self.unshuffle(x)
        n = x.shape[1]
        a, c, b, d = (p[:, 0::4], p[:, 1::4], p[:, 2::4], p[:, 3::4])
        ll = (a + b + c + d) * 0.5
        lh = (-a - b + c + d) * 0.5
        hl = (-a + b - c + d) * 0.5
        hh = (a - b - c + d) * 0.5
        return torch.cat([ll, lh, hl, hh], dim=1)

class HaarIDWT(nn.Module):
    """Exact inverse of HaarDWT."""

    def __init__(self):
        super().__init__()
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        n = x.shape[1] // 4
        ll, lh, hl, hh = (x[:, :n], x[:, n:2 * n], x[:, 2 * n:3 * n], x[:, 3 * n:])
        a = (ll - lh - hl + hh) * 0.5
        b = (ll - lh + hl - hh) * 0.5
        c = (ll + lh - hl - hh) * 0.5
        d = (ll + lh + hl + hh) * 0.5
        p = torch.stack([a, c, b, d], dim=2).flatten(1, 2)
        return self.shuffle(p)

class Net(nn.Module):
    """MULTI-LEVEL WAVELET denoiser (MWCNN family).

    Two wavelet levels, so the deepest features see a quarter of the linear resolution while
    the transform stays exactly invertible -- a property no pooling or strided-conv U-Net in
    the corpus has. It is also distinct from DenoiseUnshuffle: pixel-unshuffle is a pure
    reordering, whereas the Haar transform mixes neighbours into frequency bands, so the
    channel axis carries FREQUENCY structure rather than spatial phase.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()
        f1, f2 = (32, 48)

        def blk(cin, cout, n=2):
            layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(inplace=True, negative_slope=0.01)]
            for _ in range(n - 1):
                layers += [nn.Conv2d(cout, cout, 3, padding=1), nn.ELU(inplace=True)]
            return nn.Sequential(*layers)
        self.e1 = blk(12, f1)
        self.e2 = blk(f1 * 4, f2)
        self.mid = blk(f2, f2, 2)
        self.d2 = blk(f2, f1 * 4)
        self.d1 = blk(f1, 12)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        w1 = self.dwt(x)
        h1 = self.e1(w1)
        w2 = self.dwt(h1)
        h2 = self.e2(w2)
        h2 = self.mid(h2) + h2
        u2 = self.idwt(self.d2(h2))
        h1 = h1 + u2
        u1 = self.idwt(self.d1(h1))
        return torch.clamp(u1 + x, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def learn(self, train_data):
        self.train()
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss = self.criterion_mse(preds, clean) * 1000 + self.criterion_l1(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()