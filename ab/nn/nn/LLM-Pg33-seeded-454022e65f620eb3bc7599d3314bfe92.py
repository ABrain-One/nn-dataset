import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr", "batch", "epoch_max", "transform"}


class EluAvgBlock(nn.Module):
    """Elu block with an AvgPool2d context branch."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 5, padding=2)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ctx_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.ctx = nn.Conv2d(channels, channels, 5, padding=2)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        c = self.ctx(self.ctx_pool(out))
        c = F.interpolate(c, size=out.shape[-2:], mode="nearest")
        return self.conv2(out + c) + x





class ResidualStack(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()
        layers = [EluAvgBlock(channels)]
        for _ in range(depth - 1):
            layers.append(nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)))
        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.stack(x)


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device

        self.in_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.block0 = ResidualStack(16, 4)
        self.down0 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.block1 = ResidualStack(32, 4)
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.block2 = ResidualStack(64, 4)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bottleneck = ResidualStack(128, 4)
        self.up2 = nn.ConvTranspose2d(128, 64, 1, stride=2, output_padding=1)
        self.block3 = ResidualStack(64, 4)
        self.up1 = nn.ConvTranspose2d(64, 32, 1, stride=2, output_padding=1)
        self.block4 = ResidualStack(32, 4)
        self.up0 = nn.ConvTranspose2d(32, 16, 1, stride=2, output_padding=1)
        self.block5 = ResidualStack(16, 4)
        self.out_conv = nn.Conv2d(16, 3, 3, padding=1)

        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()

        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        e0 = self.in_conv(x)
        e0 = self.block0(e0)
        e1 = self.down0(e0)
        e1 = self.block1(e1)
        e2 = self.down1(e1)
        e2 = self.block2(e2)
        bt = self.down2(e2)
        bt = self.bottleneck(bt)
        d2 = self.block3(self.up2(bt))
        d2 = F.interpolate(d2, size=e2.shape[-2:], mode='nearest')
        d1 = self.block4(self.up1(d2))
        d1 = F.interpolate(d1, size=e1.shape[-2:], mode='nearest')
        d0 = self.block5(self.up0(d1))
        d0 = F.interpolate(d0, size=e0.shape[-2:], mode='nearest')
        return torch.clamp(self.out_conv(d0) + x, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            bad = not torch.isfinite(loss)
            if not bad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
                self.optimizer.step()
            else:
                continue
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
