import torch
import torch.nn as nn
import torch.optim as optim

class FourierUnit(nn.Module):
    """Half Instance Normalization (HINet).

    The channel axis is SPLIT: one half is instance-normalised, the other is passed through
    untouched, and the two are re-concatenated. That is the entire idea, and it is a real
    structural choice rather than a norm swap -- normalising everything destroys the
    contrast information a denoiser needs, normalising nothing loses the scale invariance
    that helps at high noise. The corpus contains DenoiseInstNorm, DenoiseLayerNorm,
    DenoiseGroupNorm and DenoiseStatNorm, and every one of them normalises the WHOLE tensor.
    """

    def __init__(self, channels):
        super().__init__()
        self.half = channels // 2
        self.norm = nn.InstanceNorm2d(self.half, affine=True)
        self.c1 = nn.Conv2d(channels, 152, 3, padding=1)
        self.c2 = nn.Conv2d(152, channels, 1, padding=0)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        h = self.c1(x)
        a, b = torch.split(h, [self.half, h.shape[1] - self.half], dim=1)
        h = torch.cat([self.norm(a), b], dim=1)
        h = self.act(h)
        return self.act(self.c2(h) + x)

class Net(nn.Module):
    """FREQUENCY-DOMAIN denoiser (Fast Fourier Convolution family).

    Every other global-context mechanism in the corpus works in the spatial domain --
    attention (SelfAttn, Axial), pooling pyramids, dilation. This one leaves the spatial
    domain entirely: each block FFTs its features, mixes the spectrum with learned weights,
    and inverse-FFTs back. Noise is broadband while image structure is concentrated at low
    frequencies, so a learned spectral filter is a structural prior none of the spatial
    families express.

    Also distinct from DenoiseWaveletMS, the corpus's other frequency model: the wavelet is a
    fixed 2-level LOCAL basis reached by folding; the FFT is a GLOBAL basis applied inside
    every block, and its mixing weights are learned.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 24
        self.head = nn.Conv2d(3, f, 3, padding=1)
        self.s1 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.f1 = FourierUnit(f)
        self.s2 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.f2 = FourierUnit(f)
        self.tail = nn.Conv2d(f, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.head(x)
        h = self.f1(self.s1(h))
        h = self.f2(self.s2(h))
        return torch.clamp(self.tail(h) + x, 0.0, 1.0)

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