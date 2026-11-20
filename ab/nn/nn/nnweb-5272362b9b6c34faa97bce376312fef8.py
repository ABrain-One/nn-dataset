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
        dropout = prm['dropout']
        
        # ResNet-like architecture adapted for CIFAR-10 (32x32 input)
        # Block 1
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.res_block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=dropout),
        )
        self.bn_res1 = nn.BatchNorm2d(64)
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.res_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=dropout),
        )
        self.bn_res2 = nn.BatchNorm2d(128)
        
        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.res_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=dropout),
        )
        self.bn_res3 = nn.BatchNorm2d(256)
        
        # Block 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.res_block4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout2d(p=dropout),
        )
        self.bn_res4 = nn.BatchNorm2d(512)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, out_shape[0])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        out = F.relu(self.bn1(self.conv1(x)))
        res = self.res_block1(out)
        out = F.relu(self.bn_res1(out + res))
        
        # Block 2
        out = F.relu(self.bn2(self.conv2(out)))
        res = self.res_block2(out)
        out = F.relu(self.bn_res2(out + res))
        
        # Block 3
        out = F.relu(self.bn3(self.conv3(out)))
        res = self.res_block3(out)
        out = F.relu(self.bn_res3(out + res))
        
        # Block 4
        out = F.relu(self.bn4(self.conv4(out)))
        res = self.res_block4(out)
        out = F.relu(self.bn_res4(out + res))
        
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out