import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# The model:
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.out_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.model = self.build_model()

    def build_model(self):
        layers = [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        # The following block is a derivative of the original ResBlock module
        # with some modifications to make it compatible with the input
        # dimensions and to follow the provided in_shape (C, H, W) channel-first format.
        class ResBlock(nn.Module):
            def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                ) if stride != 1 or in_channels != out_channels else nn.Identity()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                identity = self.downsample(x)
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                return self.relu(out + identity)

        # Add a ResBlock to the model
        layers.append(ResBlock(128, 256, stride=2))

        # Final adaptive avg pool and classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, self.out_classes)

        # Total parameter count (for the provided limit)
        self.total_params = sum(p.numel() for p in self.parameters())

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()