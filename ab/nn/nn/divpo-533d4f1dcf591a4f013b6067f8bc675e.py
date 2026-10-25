import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# This is the pre-defined configuration for the dataset
in_shape = (3, 32, 32)
out_shape = (10,)

# Defining the Net class for the image classification model
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer1 = self.make_layer(32, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])

    def make_layer(self, in_channels: int, out_channels: int, blocks: int) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data: torch.utils.data.DataLoader) -> None:
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def to_device(self, device: torch.device) -> None:
        self.device = device
        self.to(device)

    def get_input_resolution(self) -> tuple:
        # This is the pre-defined configuration for the dataset
        in_shape = (3, 32, 32)
        return in_shape

    def get_hyperparameters(self) -> list:
        return ['lr', 'momentum']

    def get_dropout(self) -> float:
        return 0.3