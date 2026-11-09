import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr'}

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        self.in_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0)
        self.down2 = nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0)
        self.res_block1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.out_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.train_setup(prm)
        self.to(device)

    def train_setup(self, prm):
        lr = prm.get('lr', 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)

    def forward(self, x):
        identity = x
        x = F.relu(self.in_conv(x))
        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        x = self.res_block1(x)
        x = F.relu(self.up2(x))
        x = F.relu(self.up1(x))
        x = torch.clamp(self.out_conv(x) + identity, 0.0, 1.0)
        return x

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss = F.mse_loss(preds, clean) * 1000 + F.l1_loss(preds, clean) * 50
            bad = not torch.isfinite(loss)
            if not bad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
                self.optimizer.step()
                total_loss += loss.item()
                count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
