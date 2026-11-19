import torch
import torch.nn.functional as F
from torch import nn, Tensor


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
        num_classes: int = out_shape[0]
        dropout: float = prm['dropout']
        
        # Stage 1: 3x3 conv, 64 channels
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Stage 2: 3x3 conv, 128 channels, stride 2 (downsampling)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Stage 3: 3x3 conv, 256 channels (Residual Block)
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3b = nn.BatchNorm2d(256)
        # 1x1 projection for skip connection (128 -> 256)
        self.conv3s = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.bn3s = nn.BatchNorm2d(256)
        
        # Stage 4: 3x3 conv, 256 channels (Residual Block)
        self.conv4a = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4b = nn.BatchNorm2d(256)
        # Identity skip connection (256 -> 256)
        
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # Stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        # Stage 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        
        # Stage 3 (Residual)
        identity = self.bn3s(self.conv3s(x))
        
        out = self.conv3a(x)
        out = self.bn3a(out)
        out = F.relu(out, inplace=True)
        out = self.conv3b(out)
        out = self.bn3b(out)
        
        out += identity
        x = F.relu(out, inplace=True)
        
        # Stage 4 (Residual)
        identity = x
        
        out = self.conv4a(x)
        out = self.bn4a(out)
        out = F.relu(out, inplace=True)
        out = self.conv4b(out)
        out = self.bn4b(out)
        
        out += identity
        x = F.relu(out, inplace=True)
        
        # Head
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x