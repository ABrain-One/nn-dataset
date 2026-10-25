import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Net class with helper class (if required)
class HelperBlock(nn.Module):
    # Non-trivial block similar to InvertedResidual or ResBlock
    def __init__(self, in_channels, out_channels, expand_ratio):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = expand_ratio == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, hidden_dim, 3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise
            nn.Conv2d(hidden_dim, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # this is the only difference compared to the template
        num_classes = out_shape[0]
        self.kernel_size = 3
        self.padding = 1
        self.stride = 1
        self.expand_ratio = 2

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            HelperBlock(32, 32, self.expand_ratio),
            nn.MaxPool2d(2, 2),
            HelperBlock(32, 64, self.expand_ratio),
            nn.MaxPool2d(2, 2),
            HelperBlock(64, 128, self.expand_ratio),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d((6, 6)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
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