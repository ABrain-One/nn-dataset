import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr", "batch", "transform"}


class DilatedBlock(nn.Module):
    """Inverted-residual (MobileNetV2-style): expand 1x1 -> depthwise -> project 1x1."""

    def __init__(self, channels):
        super().__init__()
        hidden = channels * 2
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden)
        self.project = nn.Conv2d(hidden, channels, 1)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.act(self.expand(x))
        out = self.act(self.dw(out))
        return x + self.project(out)


class LiteDenoisingBlock(nn.Module):
    """
    Lightweight denoising block with an internal channel-reduction
    bottleneck (f -> f/2 -> f), standard 3x3 convolutions, hardware-
    native ReLU activations, and a local residual connection.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.conv2 = nn.Conv2d(mid, out_channels, 3, padding=1)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.actv(self.conv1(x))
        out = self.conv2(out)
        return self.actv(out + identity)


class FeatureAttention(nn.Module):
    """Feature-wise attention mechanism."""

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Conv2d(dim, dim, 1)
        self.key = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, C, -1)
        k = self.key(x).view(B, C, -1).permute(0, 2, 1)
        v = self.value(x).view(B, C, -1)
        attn = torch.matmul(q, k).softmax(dim=-1)
        out = torch.matmul(attn, v).view(B, C, H, W)
        return out


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        Block = DilatedBlock
        Attention = FeatureAttention

        self.in_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.eb0 = Block(16)
        self.att0 = Attention(16)
        self.down0 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb1 = Block(32)
        self.att1 = Attention(32)
        self.down1 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb2 = Block(64)
        self.att2 = Attention(64)
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.bottleneck = Block(128)
        self.att_b = Attention(128)
        self.deconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.up2 = nn.Conv2d(192, 64, 3, padding=1)
        self.eb2_u = Block(64)
        self.att2_u = Attention(64)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.up1 = nn.Conv2d(96, 32, 3, padding=1)
        self.eb1_u = Block(32)
        self.att1_u = Attention(32)
        self.deconv0 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.up0 = nn.Conv2d(48, 16, 3, padding=1)
        self.eb0_u = Block(16)
        self.att0_u = Attention(16)
        self.out_conv = nn.Conv2d(16, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        e0 = self.in_conv(x)
        e0 = self.eb0(e0)
        e0 = self.att0(e0)
        e1 = self.down0(e0)
        e1 = self.eb1(e1)
        e1 = self.att1(e1)
        e2 = self.down1(e1)
        e2 = self.eb2(e2)
        e2 = self.att2(e2)
        bt = self.down2(e2)
        bt = self.bottleneck(bt)
        bt = self.att_b(bt)
        d2 = self.up2(torch.cat([self.deconv2(bt), e2], 1))
        d2 = self.eb2_u(d2)
        d2 = self.att2_u(d2)
        d1 = self.up1(torch.cat([self.deconv1(d2), e1], 1))
        d1 = self.eb1_u(d1)
        d1 = self.att1_u(d1)
        d0 = self.up0(torch.cat([self.deconv0(d1), e0], 1))
        d0 = self.eb0_u(d0)
        d0 = self.att0_u(d0)
        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

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
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
