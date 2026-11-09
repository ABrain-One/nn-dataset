import torch
import torch.nn as nn
import torch.optim as optim

def supported_hyperparameters():
    return {"lr"}

class HybridBlock(nn.Module):
    """Combines elements from ChanShuffleBlock and LBlock."""
    def __init__(self, channels):
        super().__init__()
        self.chan_shuffle = nn.ChannelShuffle(channels // 4)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.shortcut = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(self.chan_shuffle(h)))
        return h + self.shortcut(x)

class Net(nn.Module):
    """Hybrid Denoising Network combining ChanShuffleBlock and Laplacian Pyramid concepts."""
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        Block = HybridBlock
        self.in_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.down0 = nn.MaxPool2d(2)
        self.eb0 = Block(16)
        self.down1 = nn.MaxPool2d(2)
        self.eb1 = Block(16)
        self.down2 = nn.MaxPool2d(2)
        self.eb2 = Block(16)
        self.bottleneck = Block(16)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.out_conv = nn.Conv2d(16, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        e0 = self.in_conv(x)
        e0 = self.eb0(e0)
        e1 = self.down0(e0)
        e1 = self.eb1(e1)
        e2 = self.down1(e1)
        e2 = self.eb2(e2)
        bt = self.down2(e2)
        bt = self.bottleneck(bt)
        d2 = self.up2(bt)
        d1 = self.up1(d2)
        d0 = self.up0(d1)
        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

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
