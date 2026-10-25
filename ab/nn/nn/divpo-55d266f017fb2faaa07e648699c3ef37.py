import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self.build_layer(32, 32, 2, stride=1)
        self.layer2 = self.build_layer(32, 64, 2, stride=2)
        self.layer3 = self.build_layer(64, 128, 2, stride=2)
        self.layer4 = self.build_layer(128, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_shape[0])
        )
        self._is_fwd_comp = False
        self._last_inner = None

    def build_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks - 1):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def compile(self):
        if self._is_fwd_comp: return self
        self._is_fwd_comp = True
        self.avgpool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=(6, 6)),
            torch.nn.Dropout(p=0.0, inplace=False)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)
            ),
            torch.nn.BatchNorm2d(
                num_features=256,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            torch.nn.ReLU(inplace=True)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)
            ),
            torch.nn.BatchNorm2d(
                num_features=128,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            torch.nn.ReLU(inplace=True)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)
            ),
            torch.nn.BatchNorm2d(
                num_features=64,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            torch.nn.ReLU(inplace=True)
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            torch.nn.BatchNorm2d(
                num_features=32,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            torch.nn.ReLU(inplace=True)
        )
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            torch.nn.BatchNorm2d(
                num_features=32,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2)
            )
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(
                in_features=256 * 6 * 6,
                out_features=1024,
                bias=True
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(
                in_features=1024,
                out_features=10,
                bias=True
            )
        )
        self._last_inner = None
        return self.layer1