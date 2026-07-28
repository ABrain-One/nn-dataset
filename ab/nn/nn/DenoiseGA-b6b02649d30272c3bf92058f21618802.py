import torch
import torch.nn as nn
import torch.optim as optim

class SPBlock(nn.Module):
    """Grouped conv + ChannelShuffle + Hardswish residual block."""

    def __init__(self, channels):
        super().__init__()
        g = 4 if channels % 4 == 0 else 1
        self.conv1 = nn.Conv2d(channels, channels, 1, padding=0, groups=g)
        self.shuffle = nn.ChannelShuffle(g)
        self.conv2 = nn.Conv2d(channels, channels, 5, padding=2)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        h = self.act(self.conv1(x))
        return self.conv2(self.shuffle(h)) + x

class Net(nn.Module):
    """SUB-PIXEL (FFDNet-style) denoiser — NOT a U-Net. The only resolution change is a
    single PixelUnshuffle at the input and PixelShuffle at the output; the body is a FLAT
    stack of residual blocks at reduced resolution. No encoder/decoder, no cross-scale skips."""

    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 48
        self.unshuffle = nn.PixelUnshuffle(2)
        self.in_conv = nn.Conv2d(3 * 4, f, 3, padding=1)
        self.b0 = SPBlock(f)
        self.b1 = SPBlock(f)
        self.b2 = SPBlock(f)
        self.b3 = SPBlock(f)
        self.b4 = SPBlock(f)
        self.b5 = SPBlock(f)
        self.out_conv = nn.Conv2d(f, 3 * 4, 5, padding=2)
        self.shuffle = nn.PixelShuffle(2)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        h = self.in_conv(self.unshuffle(x))
        h = self.b0(h)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        h = self.b4(h)
        h = self.b5(h)
        return torch.clamp(self.shuffle(self.out_conv(h)) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        bn_layers = [m for m in self.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))]
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
            self.optimizer.zero_grad()
            bn_state = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone()) for m in bn_layers]
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            bad = not torch.isfinite(loss)
            if not bad and bn_layers:
                bad = not all((torch.isfinite(m.running_mean).all() and torch.isfinite(m.running_var).all() for m in bn_layers))
            if bad:
                for m, (rm, rv, nb) in zip(bn_layers, bn_state):
                    m.running_mean.copy_(rm)
                    m.running_var.copy_(rv)
                    m.num_batches_tracked.copy_(nb)
                continue
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)