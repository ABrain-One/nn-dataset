import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Standard import of nn modules
import torch.nn as nn
import torch.nn.functional as F

# Ensure no pretrained weights are used
def allowed_modules(*args):
    return args

# Ensure no pretrained weights are used
def disallowed_modules(*args):
    return ['models.resnet.resnet', 'models.resnet3d.resnet', 'models.resnet.ibn_resnet', 'models.resnet.augresnet', 'models.resnet.resnet_34', 'models.resnet.resnet_50', 'models.resnet.resnet_101', 'models.resnet.resnet_152']

# This is the auto-generated compatible import block
# NOTE: This is the only part that may be outside the standard nn module
#       Must be inside the class to avoid circular imports
# ENABLED: import models.resnet, models.resnet3d, models.resnet.ibn_resnet, models.resnet.augresnet, models.resnet.resnet_34, models.resnet.resnet_50, models.resnet.resnet_101, models.resnet.resnet_152
# DISABLED: import timm.models.resnet, timm.models.resnet_dla, timm.models.resnet_cs, timm.models.resnet_ibn, timm.models.resnet_rsb, timm.models.resnet_x, timm.models.resnet_2plus1d, timm.models.resnet_csb, timm.models.resnet_csb2, timm.models.resnet_csb3, timm.models.resnet_csb4, timm.models.resnet_csb5, timm.models.resnet_csb6, timm.models.resnet_csb7, timm.models.resnet_csb8

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)

        class Block(nn.Module):
            def __init__(self, in_channels: int) -> None:
                super().__init__()
                self.block = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.block(x)

        self.block = Block(32)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0)
        )

    def learn(self, train_data: torch.utils.data.DataLoader) -> None:
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)