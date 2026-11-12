import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 64),
            Block(64, 128)
        )
        self.to_reduce_params = nn.Sequential(
            Block(128, 128),
            Block(128, 64),
            Block(64, 32),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.3))
        self.flatten = nn.Flatten()
        self._calc_dims(in_shape)
        self.classifier = nn.Linear(self.flat_dim, num_classes)

    def _calc_dims(self, in_shape):
        # Calculate and store input dimension after layers for convenience
        x = torch.zeros(in_shape)
        for layer in self.stem:
            x = layer(x)
        x = self.to_reduce_params(x)
        self.flat_dim = x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        x = self.stem(x)
        x = self.to_reduce_params(x)
        x = self.dropout(x)
        x = self.flatten(x)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

    def count_params(self):
        return super().count_params() <= 500000