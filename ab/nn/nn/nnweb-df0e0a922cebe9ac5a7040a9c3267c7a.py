import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'dropout_aux'}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


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
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        num_classes = out_shape[0]
        dropout = prm['dropout']
        dropout_aux = prm['dropout_aux']
        
        # Input shape is (batch, channels, height, width)
        in_channels = in_shape[1]
        
        # Stage 1: 64 channels
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: 128 channels, stride 2 (downsampling)
        self.stage2 = ResidualBlock(64, 128, stride=2)
        
        # Stage 3: 256 channels, stride 2 (downsampling)
        self.stage3 = ResidualBlock(128, 256, stride=2)
        
        # Stage 4: 256 channels, stride 1 (no downsampling, keep depth)
        self.stage4 = ResidualBlock(256, 256, stride=1)
        
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(256, num_classes)
        
        # Use dropout_aux to satisfy hyperparameter requirement, though this arch doesn't have aux heads.
        # We can apply it to the classifier input or just store it. 
        # To strictly "use" it, let's apply it to the features before classification as a regularization.
        self.dropout_aux = nn.Dropout(p=dropout_aux)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.dropout_aux(x)
        x = self.classifier(x)
        return x