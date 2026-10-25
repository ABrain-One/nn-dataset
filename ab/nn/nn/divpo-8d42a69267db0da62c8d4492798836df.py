import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.out_channels = 32
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(self.out_channels * 6 * 6, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to_device(self, device: torch.device) -> None:
        self.to(device)

    def get_hyperparameters(self) -> dict:
        return {'lr': self.learning_rate, 'momentum': self.momentum}

    def get_input_resolution(self) -> tuple:
        # Use (64, 3, 32, 32) to meet edge constraints
        return (64, 3, 32, 32)

    def get_output_shape(self) -> tuple:
        return torch.Size([self.num_classes])

    def get_device(self) -> torch.device:
        return self.device

    def get_train_dataset(self) -> list[torch.Tensor]:
        # Generate a dummy dataset for training
        batch_size = 64
        channels, height, width = 3, 32, 32
        inputs = torch.randn(batch_size, channels, height, width, device=self.device)
        labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        return [(inputs, labels)]

    def get_eval_dataset(self) -> list[torch.Tensor]:
        # Generate a dummy dataset for evaluation
        batch_size = 64
        channels, height, width = 3, 32, 32
        inputs = torch.randn(batch_size, channels, height, width, device=self.device)
        labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        return [(inputs, labels)]