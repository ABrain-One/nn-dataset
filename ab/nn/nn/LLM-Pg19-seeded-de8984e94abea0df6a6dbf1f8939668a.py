import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChanShuffleBlock(nn.Module):
    """Grouped conv + ChannelShuffle + Hardswish residual block."""

    def __init__(self, channels):
        super().__init__()
        g = 4 if channels % 4 == 0 else 1
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, groups=g)
        self.shuffle = nn.ChannelShuffle(g)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        h = self.act(self.conv1(x))
        return self.conv2(self.shuffle(h)) + x


def supported_hyperparameters():
    return {'lr'}

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=in_shape[1], out_channels=32, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.act2 = nn.ReLU(inplace=True)
        self.chan_shuffle_block = ChanShuffleBlock(64)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.act3 = nn.ReLU(inplace=True)
        self.bottleneck = nn.Conv2d(128, 64, 3, padding=1)
        self.act4 = nn.ReLU(inplace=True)
        self.up2 = nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1, output_padding=1)
        self.act5 = nn.ReLU(inplace=True)
        self.up1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.act6 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 3, 3, padding=1)
        self.train_setup(prm)
        self.to(device)

    def train_setup(self, prm):
        lr = prm.get('lr', 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.act1(out)
        out = self.down1(out)
        out = self.act2(out)
        out = self.chan_shuffle_block(out)
        out = self.down2(out)
        out = self.act3(out)
        out = self.bottleneck(out)
        out = self.act4(out)
        out = self.up2(out)
        out = self.act5(out)
        out = self.up1(out)
        out = self.act6(out)
        out = self.conv2(out)
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
        return total_loss / len(train_data)
