import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Note: in_shape is (N, C, H, W) for this dataset (N is batch, C is channels, H/W are image dimensions).
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.out_shape = out_shape
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

        self.model = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, self.out_shape[0])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example of how to use Net with in_shape:
# Given in_shape is (N, C, H, W), we should use C, H, W for input dimensions.
# For example, for CIFAR-10, C is 3, H is 32, W is 32.
# Placeholder for a possible::__init__:
def __init__(self):
    self.net = Net(in_shape=(3, 32, 32), out_shape=(10,), prm={'lr': 0.01, 'momentum': 0.9}, device=torch.device('cpu'))
    print(f"Total params: {count_parameters(self.net)}")