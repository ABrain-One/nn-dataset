import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class LPPoolBlock(nn.Module):
    """LPPool2d context branch + Hardsigmoid gate."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.pool = nn.LPPool2d(2, 4)
        self.ctx = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Hardsigmoid(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        h = self.conv1(x)
        c = self.ctx(self.pool(h))
        c = nn.functional.interpolate(c, size=h.shape[-2:], mode="nearest")
        return self.conv2(h * self.gate(c)) + x


class SplitNode(nn.Module):
    """One encoder/decoder node: split channels, conv each half down, recurse (or mid at leaf),
    concat siblings, transpose up, add skip. Distinct weights per node => true multi-path tree."""

    def __init__(self, f, level, depth):
        super().__init__()
        self.half = f // 2
        self.pad = nn.ReflectionPad2d(1)
        self.conv_a = nn.Conv2d(self.half, f, 3, stride=2)
        self.conv_b = nn.Conv2d(self.half, f, 3, stride=2)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.leaf = level >= depth - 1
        if self.leaf:
            self.mid_a = LPPoolBlock(f)
            self.mid_b = LPPoolBlock(f)
        else:
            self.child_a = SplitNode(f, level + 1, depth)
            self.child_b = SplitNode(f, level + 1, depth)
        self.up = nn.ConvTranspose2d(2 * f, f, 2, stride=2)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        skip = x
        a = self.act(self.conv_a(self.pad(x[:, :self.half])))
        b = self.act(self.conv_b(self.pad(x[:, self.half:])))
        if self.leaf:
            a = self.mid_a(a); b = self.mid_b(b)
        else:
            a = self.child_a(a); b = self.child_b(b)
        m = self.act2(self.up(torch.cat([a, b], dim=1)))
        return m + skip


class ResidualBlock(nn.Module):
    """Basic residual block with convolutional layers and ReLU activation."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.act(self.conv1(x))) + x)


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        f = 32                       # Number of filters
        self.head = nn.Conv2d(ch, f, 3, padding=1)
        self.tree = SplitNode(f, 0, depth=4)
        self.tail = nn.Conv2d(f, ch, 3, padding=1)
        self.res_block = ResidualBlock(f)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        x = self.head(x)
        x = self.tree(x)
        x = self.res_block(x)
        return torch.clamp(self.tail(x) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)

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
