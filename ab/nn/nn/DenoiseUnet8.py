import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class EdgeFrequencyLoss(nn.Module):
    """
    L1 pixel loss + finite-difference edge loss + FFT magnitude loss.
    Combines spatial and frequency penalties for sharp, artefact-free output.
    """
    def __init__(self, edge_weight=0.1, freq_weight=0.05):
        super().__init__()
        self.edge_weight = edge_weight
        self.freq_weight = freq_weight

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        diff = pred - target
        edge = (diff[:, :, :, 1:] - diff[:, :, :, :-1]).abs().mean() + \
               (diff[:, :, 1:, :] - diff[:, :, :-1, :]).abs().mean()
        pf = torch.fft.rfft2(pred, norm='ortho').abs()
        tf = torch.fft.rfft2(target, norm='ortho').abs()
        freq = F.l1_loss(pf, tf)
        return l1 + self.edge_weight * edge + self.freq_weight * freq


class _ChanGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x * torch.sigmoid(self.conv(F.adaptive_avg_pool2d(x, 1)))


class _SpatGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 3)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        attn = F.pad(torch.cat([avg, mx], dim=1), (1, 1, 1, 1), mode='reflect')
        return x * torch.sigmoid(self.conv(attn))


class _FusionBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3)
        self.conv2 = nn.Conv2d(filters, filters, 3)
        self.chan_gate = _ChanGate(filters)
        self.spat_gate = _SpatGate()
        self.act = nn.GELU()

    def forward(self, x):
        net = self.act(self.conv1(F.pad(x, (1, 1, 1, 1), mode='reflect')))
        net = self.chan_gate(net)
        net2 = net + x
        net = self.act(self.conv2(F.pad(net2, (1, 1, 1, 1), mode='reflect')))
        net = self.spat_gate(net)
        return net + net2


class BinaryTreeFusionNet(nn.Module):
    def __init__(self, num_filters=32):
        super().__init__()
        f = num_filters
        h = f // 2

        self.init_conv = nn.Conv2d(3, f, 3, 1, 1)
        self.act = nn.GELU()

        self.enc1 = nn.ModuleList([nn.Conv2d(h, f, 3, 2) for _ in range(2)])
        self.enc2 = nn.ModuleList([nn.Conv2d(h, f, 3, 2) for _ in range(4)])
        self.enc3 = nn.ModuleList([nn.Conv2d(h, f, 3, 2) for _ in range(8)])
        self.enc4 = nn.ModuleList([nn.Conv2d(h, f, 3, 2) for _ in range(16)])

        self.fuse_blocks = nn.ModuleList([_FusionBlock(f) for _ in range(16)])

        kw = dict(stride=2, padding=1, output_padding=1)
        self.dec4 = nn.ModuleList([nn.ConvTranspose2d(f * 2, f, 3, **kw) for _ in range(8)])
        self.dec3 = nn.ModuleList([nn.ConvTranspose2d(f * 2, f, 3, **kw) for _ in range(4)])
        self.dec2 = nn.ModuleList([nn.ConvTranspose2d(f * 2, f, 3, **kw) for _ in range(2)])
        self.dec1 = nn.ConvTranspose2d(f * 2, f, 3, **kw)

        self.final_conv = nn.Conv2d(f, 3, 3, 1, 1)

    def _enc_conv(self, x, conv):
        return self.act(conv(F.pad(x, (1, 1, 1, 1), mode='reflect')))

    def _dec_blk(self, a, b, skip, conv_t):
        return self.act(conv_t(torch.cat([a, b], dim=1))) + skip

    def forward(self, inp):
        x = self.init_conv(inp)

        c1 = self._enc_conv(x[:, :x.shape[1] // 2], self.enc1[0])
        c2 = self._enc_conv(x[:, x.shape[1] // 2:], self.enc1[1])

        stage2 = []
        for i, c in enumerate([c1, c2]):
            stage2.append(self._enc_conv(c[:, :c.shape[1] // 2], self.enc2[2 * i]))
            stage2.append(self._enc_conv(c[:, c.shape[1] // 2:], self.enc2[2 * i + 1]))
        c3, c4, c5, c6 = stage2

        stage3 = []
        for i, c in enumerate([c3, c4, c5, c6]):
            stage3.append(self._enc_conv(c[:, :c.shape[1] // 2], self.enc3[2 * i]))
            stage3.append(self._enc_conv(c[:, c.shape[1] // 2:], self.enc3[2 * i + 1]))

        stage4 = []
        for i, c in enumerate(stage3):
            stage4.append(self._enc_conv(c[:, :c.shape[1] // 2], self.enc4[2 * i]))
            stage4.append(self._enc_conv(c[:, c.shape[1] // 2:], self.enc4[2 * i + 1]))

        stage4 = [self.fuse_blocks[i](stage4[i]) for i in range(16)]

        fuse4 = [self._dec_blk(stage4[2 * i], stage4[2 * i + 1], stage3[i], self.dec4[i]) for i in range(8)]
        fuse3 = [self._dec_blk(fuse4[2 * i], fuse4[2 * i + 1], [c3, c4, c5, c6][i], self.dec3[i]) for i in range(4)]
        fuse2 = [self._dec_blk(fuse3[2 * i], fuse3[2 * i + 1], [c1, c2][i], self.dec2[i]) for i in range(2)]
        fuse1 = self._dec_blk(fuse2[0], fuse2[1], x, self.dec1)

        return inp + self.final_conv(fuse1)


class Net(nn.Module):
    """
    Binary-tree fusion denoising network that recursively splits channels
    into independent encoder branches, processes them with channel/spatial
    gated fusion blocks at the deepest level, then recombines via transposed
    convolutions. Trained with an edge + frequency loss.
    """
    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        self.net = BinaryTreeFusionNet(num_filters=32)
        self.criterion = EdgeFrequencyLoss()

        self.train_setup(prm)
        self.to(self.device)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5
        )

    def forward(self, x):
        return torch.clamp(self.net(x), 0.0, 1.0)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()

            preds = self(noisy)
            loss = self.criterion(preds, clean)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()

            total_loss += loss.item()
            count += 1

        self.scheduler.step()
        return total_loss / max(count, 1)
