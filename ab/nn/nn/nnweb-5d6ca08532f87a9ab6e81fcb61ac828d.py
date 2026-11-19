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
        
        # Wide ResNet-style architecture for fast convergence
        # Using wider channels (128) helps with feature learning in 1 epoch
        
        def conv_block(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        
        # Block 1
        self.conv1 = nn.Conv2d(in_shape[1], 128, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 2, bias=False),
            nn.BatchNorm2d(128),
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(128, 256, 1, 2, bias=False),
            nn.BatchNorm2d(256),
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.downsample4 = nn.Sequential(
            nn.Conv2d(256, 512, 1, 2, bias=False),
            nn.BatchNorm2d(512),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, out_shape[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        
        # Residual Block 2
        identity = self.downsample2(x)
        out = self.block2(x)
        x = out + identity
        
        # Residual Block 3
        identity = self.downsample3(x)
        out = self.block3(x)
        x = out + identity
        
        # Residual Block 4
        identity = self.downsample4(x)
        out = self.block4(x)
        x = out + identity
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x