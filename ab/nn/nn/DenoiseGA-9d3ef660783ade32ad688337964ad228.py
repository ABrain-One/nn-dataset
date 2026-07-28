import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class DilatedBlock(nn.Module):
    """Multi-dilation residual block (1,2,4)."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        return self.act(self.conv3(out) + x)


class Net(nn.Module):
    """PoolDeconv skeleton variant — crossover-capable channel-preserving block."""

    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        Block = DilatedBlock
        self.in_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.eb0_0 = Block(16)
        self.pool0 = nn.MaxPool2d(2)
        self.down0 = nn.Conv2d(16, 32, 3, padding=1)
        self.eb1_0 = Block(32)
        self.pool1 = nn.MaxPool2d(2)
        self.down1 = nn.Conv2d(32, 64, 3, padding=1)
        self.eb2_0 = Block(64)
        self.pool2 = nn.MaxPool2d(2)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bottleneck = Block(128)
        self.deconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.up2 = nn.Conv2d(192, 64, 3, padding=1)
        self.db2_0 = Block(64)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.up1 = nn.Conv2d(96, 32, 3, padding=1)
        self.db1_0 = Block(32)
        self.deconv0 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.up0 = nn.Conv2d(48, 16, 3, padding=1)
        self.db0_0 = Block(16)
        self.out_conv = nn.Conv2d(16, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        e0 = self.in_conv(x)
        e0 = self.eb0_0(e0)
        e1 = self.down0(self.pool0(e0))
        e1 = self.eb1_0(e1)
        e2 = self.down1(self.pool1(e1))
        e2 = self.eb2_0(e2)
        bt = self.down2(self.pool2(e2))
        bt = self.bottleneck(bt)
        d2 = self.up2(torch.cat([self.deconv2(bt), e2], 1))
        d2 = self.db2_0(d2)
        d1 = self.up1(torch.cat([self.deconv1(d2), e1], 1))
        d1 = self.db1_0(d1)
        d0 = self.up0(torch.cat([self.deconv0(d1), e0], 1))
        d0 = self.db0_0(d0)
        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5
        )

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
                # BN buffers were updated by this batch's forward BEFORE the
                # check — restore them or eval is poisoned while train stays ok
                for m, (rm, rv, nb) in zip(bn_layers, bn_state):
                    m.running_mean.copy_(rm)
                    m.running_var.copy_(rv)
                    m.num_batches_tracked.copy_(nb)
                continue
            if not torch.isfinite(loss): continue  # NaN-guard (yield>90%)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
