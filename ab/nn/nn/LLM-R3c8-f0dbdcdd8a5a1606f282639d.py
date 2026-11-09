import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}

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


class LiteDenoisingBlock(nn.Module):
    """
    Lightweight denoising block with an internal channel-reduction
    bottleneck (f -> f/2 -> f), standard 3x3 convolutions, hardware-
    native ReLU activations, and a local residual connection.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.conv2 = nn.Conv2d(mid, out_channels, 3, padding=1)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.actv(self.conv1(x))
        out = self.conv2(out)
        return self.actv(out + identity)


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        channels = in_shape[1]
        f = 16 * channels
        self.head = nn.Conv2d(channels, f, 3, padding=1)
        self.s1 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.f1 = FourierUnit(f)
        self.s2 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.f2 = FourierUnit(f)
        self.body = nn.Sequential(LiteDenoisingBlock(f, f), MDTA(f, heads=2))
        self.tail = nn.Conv2d(f, channels, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.head(x)
        h = self.f1(self.s1(h))
        h = self.f2(self.s2(h))
        h = self.body[0](h)
        h = self.body[1](h)
        return torch.clamp(self.tail(h) + x, 0.0, 1.0)

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
            loss = self.criterion_mse(preds, clean) * 1000 + self.criterion_l1(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)