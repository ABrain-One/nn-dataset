import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        dropout: float = prm['dropout']
        
        # Stage 1: 3 -> 64 channels
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Stage 2: 64 -> 128 channels, with residual skip
        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn2b = nn.BatchNorm2d(128)
        # Residual connection: input is 64x32x32, output of conv2b is 128x32x32
        # We need to project the input to match dimensions for the skip connection
        self.residual_proj = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128)
        )
        
        # Stage 3: 128 -> 256 channels, with residual skip and downsampling
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3b = nn.BatchNorm2d(256)
        # Residual connection: input is 128x32x32, output of conv3b is 256x16x16
        # We need to project the input to match dimensions (128->256, 32->16)
        self.residual_proj2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )
        
        # Stage 4: 256 -> 256 channels (reaching up to 256 channels)
        # The description says "4 convolution stages reaching up to 256 channels".
        # Usually this implies the final stage maintains or increases to the max width.
        # Let's add a 4th stage with 256 channels.
        self.conv4a = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4b = nn.BatchNorm2d(256)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Calculate input size for linear layer based on in_shape
        # CIFAR-10 is 32x32.
        # Stage 1: 32x32
        # Stage 2: 32x32 (no downsample)
        # Stage 3: 16x16 (stride 2)
        # Stage 4: 16x16 (no downsample)
        # So the feature map is 256 x 16 x 16
        # However, to be robust to in_shape, we should derive the spatial size.
        # Let's assume standard CIFAR-10 32x32 for the calculation logic if we don't use AdaptiveAvgPool.
        # But the prompt says "Never hard-code the input resolution; derive what you need from in_shape."
        # So we must calculate the spatial dimensions dynamically.
        
        # Let's trace spatial dimensions:
        # Input: H, W
        # Conv1 (3x3, pad 1): H, W
        # Conv2 (3x3, pad 1): H, W
        # Conv3 (3x3, stride 2, pad 1): H//2, W//2
        # Conv4 (3x3, pad 1): H//2, W//2
        
        h = in_shape[2]
        w = in_shape[3]
        h = (h + 2*1 - 3) // 1 + 1 # Conv1
        w = (w + 2*1 - 3) // 1 + 1
        h = (h + 2*1 - 3) // 1 + 1 # Conv2
        w = (w + 2*1 - 3) // 1 + 1
        h = (h + 2*1 - 3) // 2 + 1 # Conv3
        w = (w + 2*1 - 3) // 2 + 1
        h = (h + 2*1 - 3) // 1 + 1 # Conv4
        w = (w + 2*1 - 3) // 1 + 1
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * h * w, out_shape[0])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        # Stage 2 with residual
        identity = self.residual_proj(x)
        
        out = self.conv2a(x)
        out = self.bn2a(out)
        out = F.relu(out, inplace=True)
        out = self.conv2b(out)
        out = self.bn2b(out)
        out += identity
        x = F.relu(out, inplace=True)
        
        # Stage 3 with residual and downsampling
        identity = self.residual_proj2(x)
        
        out = self.conv3a(x)
        out = self.bn3a(out)
        out = F.relu(out, inplace=True)
        out = self.conv3b(out)
        out = self.bn3b(out)
        out += identity
        x = F.relu(out, inplace=True)
        
        # Stage 4
        out = self.conv4a(x)
        out = self.bn4a(out)
        out = F.relu(out, inplace=True)
        out = self.conv4b(out)
        out = self.bn4b(out)
        x = F.relu(out, inplace=True)
        
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x