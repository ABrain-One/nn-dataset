import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Conv2d(in_channels=in_shape[1], out_channels=32, kernel_size=3, padding=1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

# Generate a unique, valid module with the given in_shape and out_shape.
# out_shape is (N, C, H, W) or (C, H, W) for CIFAR-10.
# The resulting module must:
# - use in_shape[1] as the input channel count (e.g., in_channels=in_shape[1])
# - have total parameter count <= 500000
# - not use pretrained weights
# - use standard layers only
# - not use any of these namespaces: {torchvision, torch, nn.modules.lazy, nn.quantized, nn.quantized.dynamic}
# - have a learn method with the signature: def learn(self, train_data)
# - use prm['dropout'] if present (e.g., self.dropout = nn.Dropout(p=prm['dropout']))
# - have at most 5 blocks of your design (e.g., one of: BasicBlock, ResBlock, InvertedResidual, DenseBlock, etc.)
# - have at most 5 blocks total (including the last one, if it's part of an efficient module)
# - derive from one of: nn.Module, nn.Cell, nn.Functional
# - have a supported_hyperparameters method that returns {'lr', 'momentum'}
# - have a forward method with the signature: def forward(self, x)
# - use prm.get('momentum', 0.9) for the optimizer
# - have a train_setup method with the signature: def train_setup(self, prm)
# - have no require_grad or inplace operations (no .requires_grad=True, no .flip(inplace=True), etc.)
# - use the given out_shape for the classifier (no hardcoded [128, 10] or similar)
# - derive from one of: nn.Module, nn.Cell, nn.Functional
# - have at most 5 blocks total (including the last one, if it's part of an efficient module)
# - not reuse any blocks from the previous best model or any shown examples