import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 32
        self.conv1 = nn.Conv2d(in_channels=in_shape[1], out_channels=f, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.down1 = nn.Conv2d(f, f*2, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(f*2, f*4, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(f*4, f*8, 3, stride=2, padding=1)
        self.bottleneck = nn.Conv2d(f*8, f*8, 3, padding=1)
        self.up3 = nn.ConvTranspose2d(f*8, f*4, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, 3, stride=2, padding=1, output_padding=1)
        self.up1 = nn.ConvTranspose2d(f*2, f, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(f, in_shape[1], kernel_size=3, padding=1)
        self.train_setup(prm)
        self.to(device)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)

    def forward(self, x):
        identity = x
        h1 = self.relu(self.conv1(x))
        h2 = self.relu(self.down1(h1))
        h3 = self.relu(self.down2(h2))
        h4 = self.relu(self.down3(h3))
        hb = self.relu(self.bottleneck(h4))
        u4 = self.relu(self.up3(hb))
        u3 = self.relu(self.up2(u4 + h3))
        u2 = self.relu(self.up1(u3 + h2))
        u1 = self.relu(self.conv2(u2 + h1))
        return torch.clamp(u1 + identity, 0.0, 1.0)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss.item()
        self.scheduler.step()
        return total_loss / len(train_data)
