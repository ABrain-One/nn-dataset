import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}


def get_block(in_channels, out_channels, stride=1, groups=1, expand_ratio=4):
    """ ConvNormActivation """
    hidden_channels = int(in_channels * expand_ratio)
    # NOTE: expand_ratio is not used as it is not supported in the current block implementation
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(hidden_channels),
        nn.ReLU6(inplace=True),
        nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        layers = []
        in_channels = 32
        for i in range(3):
            layers.append(get_block(in_channels, in_channels, groups=in_channels if i == 0 else 1, expand_ratio=4))
        self.layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(32 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
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
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()