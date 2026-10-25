import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Net class with helper class Block (if not provided)
class Block(nn.Module):
    # Simple Block with BatchNorm for MobileNetV3-like inverted residual
    def __init__(self, in_channels, out_channels, stride, expansion):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.use_res_conn = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return x + self.conv(x) if self.use_res_conn else self.conv(x)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.kernel_size = 3
        self.padding = 1
        self.dropout = prm.get('dropout', 0.2)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32, 1, 3),
            Block(32, 32, 2, 3),
            nn.Dropout(p=self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32, 64, 1, 3),
            Block(64, 64, 2, 3),
            nn.Dropout(p=self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(64, 128, 1, 3),
            Block(128, 128, 2, 3),
            nn.Dropout(p=self.dropout),
        )

        mid_channels = 128
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(mid_channels * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

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