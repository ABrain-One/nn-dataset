import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'batch', 'epoch_max', 'transform'}


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        self.down = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, in_shape[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1),
        )
        self.residual_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.scale = nn.Parameter(torch.tensor([0.1]))
        self.train_setup(prm)
        self.to(device)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=prm.get('epoch_max', 5))

    def forward(self, x):
        x_down = self.down(x)
        x_res = self.residual_block(x_down)
        x_up = self.up(x_res)
        x_out = self.scale * x_up + x
        return torch.clamp(x_out, 0, 1)

    def learn(self, train_data):
        self.train()
        total_loss = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            denoised = self(noisy)
            loss = F.mse_loss(denoised, clean)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        self.scheduler.step()
        return total_loss / len(train_data)
