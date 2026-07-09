import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class EdgePreservingLoss(nn.Module):
    """L1 pixel loss + L1 on spatial gradient magnitude (Sobel-like finite diff)."""
    def __init__(self, edge_weight=0.1):
        super().__init__()
        self.edge_weight = edge_weight

    def _grad_mag(self, x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx.abs().mean() + dy.abs().mean()

    def forward(self, pred, target):
        return F.l1_loss(pred, target) + self.edge_weight * (
            self._grad_mag(pred - target)
        )


class LightFusionNet(nn.Module):
    def __init__(self, enc_blocks, dec_blocks, bottom_layers, num_filters):
        super().__init__()
        f = num_filters

        self.enc_layers = nn.ModuleList()
        self.pools = nn.ModuleList()
        self._enc_channels = []
        in_ch = 3

        for n_blks in enc_blocks:
            group = nn.ModuleList()
            for k in range(n_blks):
                group.append(nn.Sequential(
                    nn.Conv2d(in_ch if k == 0 else f, f, 3, padding=1),
                    nn.ELU(),
                ))
                in_ch = f
            self.enc_layers.append(group)
            self._enc_channels.append(f)
            self.pools.append(nn.AvgPool2d(2))
            f = f * 2

        self.bottom = nn.ModuleList()
        for k in range(bottom_layers):
            self.bottom.append(nn.Sequential(
                nn.Conv2d(in_ch if k == 0 else f, f, 3, padding=1),
                nn.ELU(),
            ))
            in_ch = f

        self.ups = nn.ModuleList()
        self.dec_layers = nn.ModuleList()

        for i, n_blks in enumerate(dec_blocks):
            skip_ch = self._enc_channels[-(i + 1)]
            self.ups.append(nn.Upsample(scale_factor=2, mode='nearest'))
            f = f // 2
            cat_ch = f * 2 + skip_ch
            group = nn.ModuleList()
            for k in range(n_blks):
                group.append(nn.Sequential(
                    nn.Conv2d(cat_ch if k == 0 else f, f, 3, padding=1),
                    nn.ELU(),
                ))
            self.dec_layers.append(group)

        self.final_conv = nn.Conv2d(f, 3, 3, padding=1)

    def forward(self, inp):
        x = inp
        skips = []

        for enc_group, pool in zip(self.enc_layers, self.pools):
            for blk in enc_group:
                x = blk(x)
            skips.append(x)
            x = pool(x)

        for blk in self.bottom:
            x = blk(x)

        for up, dec_group, skip in zip(self.ups, self.dec_layers, reversed(skips)):
            x = torch.cat([up(x), skip], dim=1)
            for blk in dec_group:
                x = blk(x)

        return self.final_conv(x)


class Net(nn.Module):
    """
    Lightweight fusion U-Net using ELU-activated 3x3 convolutions, average
    pooling for downsampling and nearest-neighbor upsampling with skip
    concatenation. Trained with an edge-preserving (L1 + gradient) loss.
    """
    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        self.net = LightFusionNet(
            enc_blocks=[2, 2, 2, 2], dec_blocks=[2, 2, 2, 2],
            bottom_layers=2, num_filters=32,
        )
        self.criterion = EdgePreservingLoss()

        self.train_setup(prm)
        self.to(self.device)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        t_max = prm.get("epoch_max", 50)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=t_max, eta_min=1e-5
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
            # Frozen recipe: Charbonnier loss (eps=1e-3)
            loss = torch.mean(torch.sqrt((preds - clean) ** 2 + 1e-3 ** 2))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_loss += loss.item()
            count += 1

        self.scheduler.step()
        return total_loss / max(count, 1)
