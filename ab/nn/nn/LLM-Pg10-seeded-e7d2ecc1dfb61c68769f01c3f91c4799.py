import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr'}

class DualPathBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(2)
        self.conv_ctx = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        h = self.act(self.conv1(x))
        ctx = self.avg_pool(x)
        ctx = self.conv_ctx(ctx)
        ctx = F.interpolate(ctx, size=h.shape[-2:], mode='nearest')
        h = h + ctx * self.gate(ctx)
        return self.act(self.conv2(h) + x)


class WeightedL1PSNRLoss(nn.Module):
    def __init__(self, psnr_weight=0.3):
        super().__init__()
        self.psnr_weight = psnr_weight

    def forward(self, pred, target):
        diff = (pred - target).abs()
        l1 = diff.mean()
        err_map = diff.detach().mean(dim=1, keepdim=True)
        weight = 1.0 + self.psnr_weight * err_map / (err_map.mean() + 1e-6)
        return (diff * weight).mean()


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        dim = 16

        self.intro = nn.Conv2d(ch, dim, 3, padding=1)
        self.enc0 = DualPathBlock(dim)
        self.down0 = nn.Conv2d(dim, dim*2, 3, stride=2, padding=1)
        self.enc1 = DualPathBlock(dim*2)
        self.down1 = nn.Conv2d(dim*2, dim*4, 3, stride=2, padding=1)
        self.bottleneck = DualPathBlock(dim*4)
        self.up1 = nn.ConvTranspose2d(dim*4, dim*2, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(dim*2, dim, 3, stride=2, padding=1, output_padding=1)
        self.ending = nn.Conv2d(dim, ch, 3, padding=1)

        self.criterion = WeightedL1PSNRLoss()

        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        x = self.intro(x)
        e0 = self.enc0(x)
        x = self.down0(e0)
        e1 = self.enc1(x)
        x = self.down1(e1)
        e2 = self.bottleneck(x)
        x = self.up1(e2)
        x = self.up2(e1 + x)
        x = self.ending(x)
        return torch.clamp(x + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss = self.criterion(preds, clean)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
