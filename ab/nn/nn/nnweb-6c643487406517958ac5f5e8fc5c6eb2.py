import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm

        # ICNet Architecture
        # Input: 32x32 (CIFAR-10)
        # ICInitBlock: 3 stride-2 convs -> 32/8 = 4x4
        # PSP: 1x1 convs
        # Head: 1x1 conv
        # Classification: Linear

        in_channels = in_shape[1]
        num_classes = out_shape[0]

        # ICInitBlock
        self.ic_init_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # PSP Block
        # Input channels: 64
        # Pool sizes: [1, 2, 3, 6]
        # Output channels: 64 (concat 64 + 4*16 = 128? No, standard PSPNet usually outputs same channels or half. 
        # Let's follow a standard ICNet classification head structure.
        # Often: PSP outputs C channels. Then a 1x1 conv to C.
        # Let's use 16 channels per branch, so 4*16=64. Total 64+64=128.
        # Then 1x1 conv to 128? Or 64?
        # Let's stick to a simple robust implementation.
        
        self.psp = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=size),
                nn.Conv2d(64, 16, kernel_size=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            ) for size in [1, 2, 3, 6]
        ])
        
        self.psp_fuse = nn.Sequential(
            nn.Conv2d(64 + 4*16, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # ICInitBlock
        x = self.ic_init_block(x)
        
        # PSP
        h, w = x.size()[2:]
        psp_out = []
        for psp in self.psp:
            out = psp(x)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            psp_out.append(out)
        
        x = torch.cat([x] + psp_out, dim=1)
        x = self.psp_fuse(x)
        
        # Head
        x = self.head(x)
        
        # Global Pool
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Classification
        out = self.classifier(x)
        return out

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()


def supported_hyperparameters():
    return {'lr', 'momentum'}