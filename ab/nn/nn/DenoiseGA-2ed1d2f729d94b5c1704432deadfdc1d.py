import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CharbonnierLoss(nn.Module):
    """sqrt((pred - target)^2 + eps^2): smoother than L1, more robust than L2."""

    def __init__(self, eps=0.001):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))

class _GatedAdaptiveBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.act2 = nn.GELU()
        self.conv4 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        net = self.act1(self.dw(self.conv1(x)))
        net = self.conv2(net)
        middle = x + net
        net = self.act2(self.conv3(middle))
        net = self.conv4(net)
        return middle + net

class GatedAdaptiveUNet(nn.Module):

    def __init__(self, in_ch=3, num_filters=32, enc_blocks=None, dec_blocks=None, mid_blocks=2):
        super().__init__()
        if enc_blocks is None:
            enc_blocks = [1, 1, 2, 2]
        if dec_blocks is None:
            dec_blocks = [1, 1, 1, 1]
        f = num_filters
        self.intro = nn.Conv2d(in_ch, f, 3, padding=1)
        self.enc_layers = nn.ModuleList()
        self.downs = nn.ModuleList()
        enc_channels = []
        for n in enc_blocks:
            self.enc_layers.append(nn.Sequential(*[_GatedAdaptiveBlock(f) for _ in range(n)]))
            enc_channels.append(f)
            self.downs.append(nn.Conv2d(f, f * 2, 2, stride=2))
            f *= 2
        self.mid = nn.Sequential(*[_GatedAdaptiveBlock(f) for _ in range(mid_blocks)])
        self.ups = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.dec_layers = nn.ModuleList()
        for i, n in enumerate(dec_blocks):
            skip_ch = enc_channels[-(i + 1)]
            self.ups.append(nn.Sequential(nn.Conv2d(f, f * 2, 1), nn.PixelShuffle(2)))
            f = f // 2
            self.fusions.append(nn.Conv2d(f + skip_ch, f, 1))
            self.dec_layers.append(nn.Sequential(*[_GatedAdaptiveBlock(f) for _ in range(n)]))
        self.output = nn.Conv2d(f, in_ch, 3, padding=1)

    def forward(self, inp):
        x = self.intro(inp)
        skips = []
        for enc, down in zip(self.enc_layers, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.mid(x)
        for up, fuse, dec, skip in zip(self.ups, self.fusions, self.dec_layers, reversed(skips)):
            x = up(x)
            x = fuse(torch.cat([x, skip], dim=1))
            x = dec(x)
        return inp + self.output(x)

class Net(nn.Module):
    """
    NAFNet-style gated adaptive U-Net using BatchNorm, simple channel-gate
    (multiplicative) non-linearities, depthwise convolutions, simplified
    channel attention and PixelShuffle upsampling. Trained with a
    Charbonnier loss.
    """

    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        self.net = GatedAdaptiveUNet(in_ch=3, num_filters=32, enc_blocks=[1, 1, 2, 2], dec_blocks=[1, 1, 1, 1], mid_blocks=4)
        self.criterion = CharbonnierLoss()
        self.train_setup(prm)
        self.to(self.device)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        t_max = prm.get('epoch_max', 50)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=1e-05)

    def forward(self, x):
        return torch.clamp(self.net(x), 0.0, 1.0)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss = torch.mean(torch.sqrt((preds - clean) ** 2 + 0.001 ** 2))
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)