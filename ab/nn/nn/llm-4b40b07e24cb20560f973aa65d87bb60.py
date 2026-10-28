import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.channels_config = [32, 64, 128, 256, 128, 64]
        layers = []
        in_channels = in_shape[1]
        layers += [nn.Conv2d(in_channels, self.channels_config[0], kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.channels_config[0]), nn.ReLU(inplace=True)]
        in_channels = self.channels_config[0]
        layers += [nn.Conv2d(in_channels, self.channels_config[1], kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.channels_config[1]), nn.ReLU(inplace=True)]
        in_channels = self.channels_config[1]
        layers += [nn.Conv2d(in_channels, self.channels_config[2], kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.channels_config[2]), nn.ReLU(inplace=True)]
        in_channels = self.channels_config[2]
        layers += [nn.Conv2d(in_channels, self.channels_config[3], kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.channels_config[3]), nn.ReLU(inplace=True)]
        in_channels = self.channels_config[3]
        layers += [nn.Conv2d(in_channels, self.channels_config[4], kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.channels_config[4]), nn.ReLU(inplace=True)]
        in_channels = self.channels_config[4]
        layers += [nn.Conv2d(in_channels, self.channels_config[5], kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.channels_config[5]), nn.ReLU(inplace=True)]
        in_channels = self.channels_config[5]
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        classifier_input_features = in_channels * 6 * 6
        self.classifier = nn.Sequential(nn.Dropout(p=prm['dropout']), nn.Linear(classifier_input_features, 512), nn.ReLU(inplace=True), nn.Dropout(p=prm['dropout']), nn.Linear(512, out_shape[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x