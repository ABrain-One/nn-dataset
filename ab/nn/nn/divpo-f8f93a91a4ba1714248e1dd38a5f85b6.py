import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            )
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=False))
        layers.extend([
            # dw
            nn.Conv2d(hidden_dim if expand_ratio != 1 else inp, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=False),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.features = self.build_features(in_channels, 32, prm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'] if 'lr' in prm else 0.01,
            momentum=prm['momentum'] if 'momentum' in prm else 0.9,
        )

    def build_features(self, in_channels, out_channels, prm):
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU6(inplace=False))

        # Use inverted residual blocks (e.g., MobileNetV3 blocks)
        # Example block (do NOT use this literally, generate your own):
        # layers.append(InvertedResidual(32, 32, 1, 3))

        # Generate up to 238 (given) blocks of your own choosing
        # Each block should be unique and not appear in any training code
        # Refer to the previous block as an example; its stride, channels, and expand_ratio should differ from others

        # Total parameters should not exceed 500000

        # Use nn.Sequential(*layers) to combine into a feature extractor
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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