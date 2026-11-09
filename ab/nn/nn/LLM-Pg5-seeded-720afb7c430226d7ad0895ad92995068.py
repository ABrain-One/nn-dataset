import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


class ELUAvgBlock(nn.Module):
    """ELU block with an AvgPool2d context branch."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 7, padding=3)
        self.ctx_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.ctx = nn.Conv2d(channels, channels, 7, padding=3)
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.01)

    def forward(self, x):
        out = self.act(self.conv1(x))
        c = self.ctx(self.ctx_pool(out))
        c = nn.functional.interpolate(c, size=out.shape[-2:], mode='nearest')
        return self.conv2(out + c) + x

class SPBlock(nn.Module):
    """Channel-preserving residual block (crossover contract)."""

    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.c2(self.act(self.c1(x))) + x)


def supported_hyperparameters():
    return {'lr'}

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        self.conv_in = nn.Conv2d(ch, 64, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(64, 64, kernel_size=2, stride=2)
        self.sp_block1 = SPBlock(64)
        self.down2 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.sp_block2 = SPBlock(128)
        self.bottleneck = ELUAvgBlock(128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_out = nn.Conv2d(64, ch, kernel_size=3, padding=1)
        self.train_setup(prm)
        self.to(device)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)

    def forward(self, x):
        identity = x
        x = self.conv_in(x)
        x = F.relu(x)
        x = self.down1(x)
        x = self.sp_block1(x)
        x = self.down2(x)
        x = self.sp_block2(x)
        x = self.bottleneck(x)
        x = self.up2(x)
        x = F.relu(x)
        x = self.up1(x)
        x = F.relu(x)
        x = self.conv_out(x)
        x += identity
        return torch.clamp(x, 0.0, 1.0)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        bn_layers = [m for m in self.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))]
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            bn_state = [(m.running_mean.clone(), m.running_var.clone(),
                         m.num_batches_tracked.clone()) for m in bn_layers]
            preds = self(noisy)
            loss_gt = F.mse_loss(preds, clean)
            loss = loss_gt * 1000 + F.l1_loss(preds, clean) * 50
            bad = not torch.isfinite(loss)
            if not bad and bn_layers:
                bad = not all(torch.isfinite(m.running_mean).all()
                              and torch.isfinite(m.running_var).all() for m in bn_layers)
            if bad:
                for m, (rm, rv, nb) in zip(bn_layers, bn_state):
                    m.running_mean.copy_(rm)
                    m.running_var.copy_(rv)
                    m.num_batches_tracked.copy_(nb)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
