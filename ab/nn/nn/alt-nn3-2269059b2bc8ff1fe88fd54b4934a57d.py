import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr': 0.01, 'momentum': 0.95, 'dropout': 0.4}

class BagNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = self.conv1x1_block(in_channels, out_channels)
        self.conv2 = self.conv_block(out_channels, out_channels, kernel_size, stride)
        self.conv3 = self.conv1x1_block(out_channels, out_channels, activation=False)

    @staticmethod
    def conv1x1_block(in_channels, out_channels, activation=True):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels) if activation else nn.Identity(),
            nn.ReLU(inplace=True) if activation else nn.Identity(),
        )

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size, stride):
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Net(nn.Module):
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        layers = []
        in_channels = in_shape[1]

                                                       
        layers += [
            nn.Conv2d(in_channels, 32, kernel_size=9, stride=3, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        in_channels = 32

                                 
        layers += [
            BagNetUnit(in_channels, 64, 3, 1),
            BagNetUnit(64, 128, 3, 2),
            BagNetUnit(128, 256, 3, 2)
        ]
        in_channels = 256

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

                                                   
        dropout_p = prm['dropout']
        classifier_input_features = in_channels * 6 * 6
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(classifier_input_features, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(2048, out_shape[0])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

                                            
                                                                                                                                                                                                                                                                                                                                           
addon_accuracy: 0.9724476505360184