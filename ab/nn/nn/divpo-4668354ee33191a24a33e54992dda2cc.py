import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple[int], out_shape: tuple[int], prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.blocks = nn.Sequential(
            self._block(32, 32, 1),
            self._block(32, 64, 2),
            self._block(64, 64, 1)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(64 * 7 * 7, num_classes)
        
        if 'dropout' in prm:
            self.dropout = nn.Dropout(prm['dropout'])
        else:
            self.dropout = None

    def _block(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        layers = nn.ModuleList()
        if stride != 1 or in_channels != out_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride))
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ])
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.blocks(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if self.dropout:
            x = self.dropout(x)
        
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data: torch.utils.data.DataLoader) -> None:
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)  # Gradient clipping
            self.optimizer.step()