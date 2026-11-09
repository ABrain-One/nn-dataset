import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {"lr"}

class AdaptiveLocalContextBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 5, padding=2)
        self.pool = nn.AdaptiveAvgPool2d(7)
        self.context = nn.Conv2d(channels, channels, 5, padding=2)
        self.act = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        out = self.act(self.conv1(x))
        ctx = self.context(self.pool(out))
        ctx = F.interpolate(ctx, size=out.shape[-2:], mode='nearest')
        return self.conv1(out + ctx) + x

class TriLevelAttentionUNet(nn.Module):
    def __init__(self, num_filters=16):
        super().__init__()
        f = num_filters

        self.act = nn.PReLU()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, f, 3, padding=1),
            AdaptiveLocalContextBlock(f),
            nn.Conv2d(f, f, 3, padding=1)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(f, f, 3, stride=2, padding=1),
            nn.Conv2d(f, f, 3, padding=1),
            AdaptiveLocalContextBlock(f),
            nn.Conv2d(f, f, 3, padding=1)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(f, f, 3, stride=2, padding=1),
            nn.Conv2d(f, f, 3, padding=1),
            AdaptiveLocalContextBlock(f),
            nn.Conv2d(f, f, 3, padding=1)
        )

        self.up3 = nn.ConvTranspose2d(f, f, 1, stride=2, output_padding=1)

        self.dec3 = nn.Sequential(
            nn.Conv2d(f * 2, f, 3, padding=1),
            AdaptiveLocalContextBlock(f),
            nn.Conv2d(f, f, 3, padding=1)
        )

        self.up2 = nn.ConvTranspose2d(f, f, 1, stride=2, output_padding=1)

        self.dec2 = nn.Sequential(
            nn.Conv2d(f * 2, f, 3, padding=1),
            AdaptiveLocalContextBlock(f),
            nn.Conv2d(f, f, 3, padding=1)
        )

        self.conv6 = nn.Conv2d(f, f, 3, padding=1)
        self.final_conv = nn.Conv2d(f, 3, 3, padding=1)

    def forward(self, inp):
        x1 = self.enc1(inp)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        
        x = self.up3(x3)
        x = torch.cat([x2, x], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.dec2(x)

        x = self.conv6(x)
        return torch.clamp(inp + self.final_conv(x), 0.0, 1.0)

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        self.net = TriLevelAttentionUNet(num_filters=16)
        self.criterion = nn.MSELoss()

        self.train_setup(prm)
        self.to(self.device)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)

    def forward(self, x):
        return self.net(x)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()

            preds = self(noisy)
            loss = self.criterion(preds, clean)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()

            total_loss += loss.item()
            count += 1

        self.scheduler.step()
        return total_loss / max(count, 1)
