import torch
from torch import nn, optim

def supported_hyperparameters():
    return {'lr'}

class Net(nn.Module):

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=in_shape[1], out_channels=137, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.01)
        self.conv2 = nn.Conv2d(in_channels=137, out_channels=80, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=80, out_channels=in_shape[1], kernel_size=5, padding=2)
        self.train_setup(prm)
        self.to(device)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += identity
        out = torch.clamp(out, 0, 1)
        return out

    def learn(self, train_data):
        self.train()
        total_loss = 0
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
            self.optimizer.zero_grad()
            denoised = self(noisy)
            loss = nn.MSELoss()(denoised, clean)
            if not torch.isfinite(loss):
                continue
            if not torch.isfinite(loss):
                continue
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_data)