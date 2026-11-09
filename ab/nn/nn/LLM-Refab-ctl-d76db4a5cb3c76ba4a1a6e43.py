import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {"lr"}

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 64
        self.conv1 = nn.Conv2d(in_channels=in_shape[1], out_channels=f, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.down1 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1)
        self.elu_avg_block = ELUAvgBlock(f)
        self.up1 = nn.ConvTranspose2d(f, f, kernel_size=4, stride=2, padding=1)
        self.double_dilated_block = DoubleDilatedBlock(f)
        self.out_conv = nn.Conv2d(f, in_shape[1], kernel_size=3, padding=1)
        self.train_setup(prm)
        self.to(device)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.down1(out)
        out = self.elu_avg_block(out)
        out = self.up1(out)
        out = self.double_dilated_block(out)
        out = self.out_conv(out)
        out += identity
        out = torch.clamp(out, 0, 1)
        return out

    def learn(self, train_data):
        self.train()
        total_loss = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            denoised = self(noisy)
            loss = nn.MSELoss()(denoised, clean)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        self.scheduler.step()
        return total_loss / len(train_data)

class ELUAvgBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ctx_pool = nn.AvgPool2d(4)
        self.ctx = nn.Conv2d(channels, channels, 1)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        c = self.ctx(self.ctx_pool(out))
        c = nn.functional.interpolate(c, size=out.shape[-2:], mode="nearest")
        return self.conv2(out + c) + x

class DoubleDilatedBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.proj_in = nn.Conv2d(channels, channels * 2, 1)
        self.dw1 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.sca1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.proj_out1 = nn.Conv2d(channels, channels, 1)
        self.gamma1 = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.01)
        
        self.dw2 = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.sca2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.proj_out2 = nn.Conv2d(channels, channels, 1)
        self.gamma2 = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.01)

    def forward(self, inp):
        x = self.norm1(inp)
        xp = self.proj_in(x)
        x1, x2 = xp.chunk(2, dim=1)
        x1 = self.dw1(x1)
        x1 = x1 * x2
        x1 = x1 * self.sca1(x1)
        x1 = self.proj_out1(x1)
        x1 = x1 * self.gamma1
        
        x2 = self.dw2(x2)
        x2 = x2 * x1
        x2 = x2 * self.sca2(x2)
        x2 = self.proj_out2(x2)
        x2 = x2 * self.gamma2
        
        return inp + (x1 + x2)
