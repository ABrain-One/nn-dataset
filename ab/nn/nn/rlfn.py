import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, in_shape=(3, 64, 64), out_shape=(3, 256, 256), prm=None):
        super(Net, self).__init__()
        # 1. Feature Extraction
        self.fea_conv = nn.Conv2d(3, 52, 3, 1, 1)
        
        # 2. Residual Block (The 'L' in RLFN)
        self.block = nn.Sequential(
            nn.Conv2d(52, 52, 3, 1, 1),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Conv2d(52, 52, 3, 1, 1)
        )
        
        # 3. Upsampler (Turns 64x64 into 256x256)
        self.upsampler = nn.Sequential(
            nn.Conv2d(52, 3 * (4**2), 3, 1, 1),
            nn.PixelShuffle(4) 
        )

    def forward(self, x):
        out = self.fea_conv(x)
        out = self.block(out) + out
        out = self.upsampler(out)
        return out
