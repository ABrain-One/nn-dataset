import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion_factor = expansion_factor

        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if self.expansion_factor != 1:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=in_channels * self.expansion_factor,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False
                    ),
                    nn.BatchNorm2d(in_channels * self.expansion_factor),
                    nn.ReLU6(inplace=True)
                )
            )

        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels * self.expansion_factor,
                    out_channels=in_channels * self.expansion_factor,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=in_channels * self.expansion_factor,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels * self.expansion_factor),
                nn.ReLU6(inplace=True)
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels * self.expansion_factor,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        )
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layer(x)
        else:
            return self.layer(x)

class MobileNetV2Block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride,
            expansion_factor
    ):
        super().__init__()
        self.block = nn.Sequential(
            InvertedResidual(
                in_channels,
                out_channels,
                stride,
                expansion_factor
            ),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        layers = []
        layers += [
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]

        inverted_residual_params = {
            'in_channels': 32,
            'out_channels': 32,
            'stride': 1,
            'expansion_factor': 1
        }

        inverted_residual_block_params = {
            'in_channels': 32,
            'out_channels': 32,
            'stride': 1,
            'expansion_factor': 1
        }

        layers += [
            MobileNetV2Block(
                inverted_residual_params['in_channels'],
                inverted_residual_params['out_channels'],
                inverted_residual_params['stride'],
                inverted_residual_params['expansion_factor']
            )
        ]

        inverted_residual_params['in_channels'] = inverted_residual_block_params['in_channels']
        inverted_residual_params['out_channels'] = inverted_residual_block_params['out_channels']
        inverted_residual_params['stride'] = inverted_residual_block_params['stride']
        inverted_residual_params['expansion_factor'] = inverted_residual_block_params['expansion_factor']

        inverted_residual_block = InvertedResidual(
            inverted_residual_params['in_channels'],
            inverted_residual_params['out_channels'],
            inverted_residual_params['stride'],
            inverted_residual_params['expansion_factor']
        )

        layers += [
            inverted_residual_block,
            nn.ReLU6(inplace=True)
        ]

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

        self.to(self.device)

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