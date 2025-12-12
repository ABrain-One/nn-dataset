import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + (in_channels // 2), out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Net(nn.Module):
    """
    Residual U-Net with Safety Clamping
    """
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        n_channels = in_shape[1]
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_channels)
        
        self.to(self.device)
        self._initialize_optimizer(prm)
        self.criterion = nn.MSELoss()

    def _initialize_optimizer(self, prm):
        raw_lr = prm.get('lr', 0.001)
        self.optimizer = optim.Adam(self.parameters(), lr=raw_lr)

    def forward(self, x):
        input_img = x
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        dec = self.up1(x5, x4)
        dec = self.up2(dec, x3)
        dec = self.up3(dec, x2)
        dec = self.up4(dec, x1)
        
        logits = self.outc(dec)
        
        return torch.clamp(input_img + logits, 0.0, 1.0)

    def train_setup(self, prm):
        self._initialize_optimizer(prm)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        
        for inputs, targets in train_data:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            count += 1
            
        return total_loss / count if count > 0 else 0.0
