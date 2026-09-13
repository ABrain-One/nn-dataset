import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'batch', 'epoch_max'}


class ELUAvgBlock(nn.Module):
    """InstanceNorm + PReLU residual block."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        self.conv1 = nn.Conv2d(channels, channels, 1, padding=0)
        self.conv2 = nn.Conv2d(channels, channels, 1, padding=0)
        self.act = nn.PReLU(channels)

    def forward(self, x):
        out = self.act(self.conv1(self.norm(x)))
        return self.conv2(out) + x


class SPBlock(nn.Module):
    """Channel-preserving residual block (crossover contract)."""

    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.c2(self.act(self.c1(x))) + x)


class LinearMixBlock(nn.Module):
    """Self-attention residual block (softmax + matmul — absent from the pool)."""

    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.pool = nn.MaxPool2d(4)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        qh = self.pool(self.q(x)).flatten(2)
        kh = self.pool(self.k(x)).flatten(2)
        vh = self.pool(self.v(x)).flatten(2)
        att = self.soft(torch.matmul(qh.transpose(1, 2), kh) / c ** 0.5)
        out = torch.matmul(vh, att.transpose(1, 2))
        out = out.reshape(b, c, h // 4, w // 4)
        out = nn.functional.interpolate(out, size=(h, w), mode='nearest')
        return x + self.proj(out)


class ELUAvgContextBlock(nn.Module):
    """ELU block with an AvgPool2d context branch."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 5, padding=2)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ctx_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.ctx = nn.Conv2d(channels, channels, 5, padding=2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        c = self.ctx(self.ctx_pool(out))
        c = nn.functional.interpolate(c, size=out.shape[-2:], mode='nearest')
        return self.conv2(out + c) + x


class Net(nn.Module):
    """Combined architecture using ELUAvgBlock, SPBlock, LinearMixBlock, and ELUAvgContextBlock."""

    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        Block = ELUAvgBlock
        ContextBlock = ELUAvgContextBlock
        SelfAttentionBlock = LinearMixBlock
        self.in_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.eb0 = Block(16)
        self.context0 = ContextBlock(16)
        self.down0 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.eb1 = Block(32)
        self.self_attention1 = SelfAttentionBlock(32)
        self.down1 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.eb2 = Block(64)
        self.self_attention2 = SelfAttentionBlock(64)
        self.down2 = nn.Conv2d(64, 128, 1, stride=2, padding=0)
        self.bottleneck = Block(128)
        self.sp_pre2 = nn.Conv2d(128, 512, 3, padding=1)
        self.sp_shuf2 = nn.PixelShuffle(2)
        self.sp_up2 = nn.Conv2d(192, 64, 3, padding=1)
        self.sp_db2 = SPBlock(64)
        self.sp_pre1 = nn.Conv2d(64, 256, 1)
        self.sp_shuf1 = nn.PixelShuffle(2)
        self.sp_up1 = nn.Conv2d(96, 32, 3, padding=1)
        self.sp_db1 = SPBlock(32)
        self.sp_pre0 = nn.Conv2d(32, 128, 5, padding=2)
        self.sp_shuf0 = nn.PixelShuffle(2)
        self.sp_up0 = nn.Conv2d(48, 16, 3, padding=1)
        self.sp_db0 = SPBlock(16)
        self.out_conv = nn.Conv2d(16, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        e0 = self.in_conv(x)
        e0 = self.eb0(e0)
        e0 = self.context0(e0)
        e1 = self.down0(e0)
        e1 = self.eb1(e1)
        e1 = self.self_attention1(e1)
        e2 = self.down1(e1)
        e2 = self.eb2(e2)
        e2 = self.self_attention2(e2)
        bt = self.down2(e2)
        bt = self.bottleneck(bt)
        sp_d2 = self.sp_db2(self.sp_up2(torch.cat([self.sp_shuf2(self.sp_pre2(bt)), e2], 1)))
        sp_d1 = self.sp_db1(self.sp_up1(torch.cat([self.sp_shuf1(self.sp_pre1(sp_d2)), e1], 1)))
        sp_d0 = self.sp_db0(self.sp_up0(torch.cat([self.sp_shuf0(self.sp_pre0(sp_d1)), e0], 1)))
        return torch.clamp(self.out_conv(sp_d0) + identity, 0.0, 1.0)

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
