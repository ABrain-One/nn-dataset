import torch
import torch.nn as nn


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


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
            nn.Conv2d(in_channels, 96, kernel_size=7, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        ]
        layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        in_channels = 96
        layers += [
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        in_channels = 128
        layers += [
            nn.Conv2d(in_channels, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ]
        in_channels = 384
        layers += [
            nn.Conv2d(in_channels, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ]
        in_channels = 384
        layers += [
            nn.Conv2d(in_channels, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        ]
        layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        in_channels = 192
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        classifier_input_features = in_channels * 6 * 6
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm['dropout']),
            nn.Linear(classifier_input_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm['dropout']),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, out_shape[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Chromosome used to generate this model:
# {'conv1_filters': 96, 'conv1_kernel': 7, 'conv1_stride': 4, 'conv2_filters': 128, 'conv2_kernel': 3, 'conv3_filters': 384, 'conv4_filters': 384, 'conv5_filters': 192, 'fc1_neurons': 4096, 'fc2_neurons': 2048, 'lr': 0.001, 'momentum': 0.95, 'dropout': 0.4, 'include_conv1': 1, 'include_conv2': 1, 'include_conv3': 1, 'include_conv4': 1, 'include_conv5': 1, 'pooling_type1': 'AvgPool2d', 'pooling_type2': 'MaxPool2d', 'pooling_type3': 'AvgPool2d', 'activation_type': 'ReLU', 'use_batchnorm': 1}