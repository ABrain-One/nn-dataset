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
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        ]
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        in_channels = 256
        layers += [
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        ]
        in_channels = 256
        layers += [
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        ]
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        in_channels = 256
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        classifier_input_features = in_channels * 6 * 6
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm['dropout']),
            nn.Linear(classifier_input_features, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=prm['dropout']),
            nn.Linear(4096, 3072),
            nn.LeakyReLU(inplace=True),
            nn.Linear(3072, out_shape[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Chromosome used to generate this model:
# {'conv1_filters': 32, 'conv1_kernel': 11, 'conv1_stride': 3, 'conv2_filters': 256, 'conv2_kernel': 3, 'conv3_filters': 256, 'conv4_filters': 384, 'conv5_filters': 256, 'fc1_neurons': 4096, 'fc2_neurons': 3072, 'lr': 0.01, 'momentum': 0.95, 'dropout': 0.4, 'include_conv1': 0, 'include_conv2': 1, 'include_conv3': 1, 'include_conv4': 0, 'include_conv5': 1, 'pooling_type1': 'AdaptiveMaxPool2d', 'pooling_type2': 'MaxPool2d', 'pooling_type3': 'MaxPool2d', 'activation_type': 'LeakyReLU', 'use_batchnorm': 1}