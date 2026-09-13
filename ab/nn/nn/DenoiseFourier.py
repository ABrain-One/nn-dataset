import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class FourierUnit(nn.Module):
    """Fast-Fourier-Convolution unit (FFC / LaMa / DeepRFT family): transform to the
    frequency domain, apply a learned pointwise mix to the spectrum, transform back. One
    frequency-domain multiply touches EVERY pixel, so the receptive field is global in a
    single op rather than grown layer by layer."""

    def __init__(self, channels):
        super().__init__()
        # real+imaginary interleaved as channels: C complex -> 2C real
        self.mix = nn.Conv2d(channels * 2, channels * 2, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        spec = torch.fft.rfft2(x, norm="ortho")            # (B, C, H, W/2+1) complex
        z = torch.view_as_real(spec)                       # (B, C, H, W/2+1, 2)
        b, c = z.shape[0], z.shape[1]
        fh, fw = z.shape[2], z.shape[3]
        z = z.permute(0, 1, 4, 2, 3).reshape(b, c * 2, fh, fw)
        z = self.act(self.mix(z))
        z = z.reshape(b, c, 2, fh, fw).permute(0, 1, 3, 4, 2).contiguous()
        out = torch.fft.irfft2(torch.view_as_complex(z), s=(h, w), norm="ortho")
        return out + x


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

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
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
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5)

    def learn(self, train_data):
        self.train()
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss = self.criterion_mse(preds, clean) * 1000 + \
                self.criterion_l1(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
