import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.num_layers = 2
        self.channels_list = [16, 32]
        self.num_filters_list = [32, 64]
        self.filter_size = 3
        self.dropout = prm.get('dropout', 0.0)
        self.dropout_ratio = prm.get('dropout', 0.0)
        self.dropout_rate = prm.get('dropout', 0.0)
        self.dropout_p = prm.get('dropout', 0.0)
        self.dropout_probability = prm.get('dropout', 0.0)
        self.dropout_val = prm.get('dropout', 0.0)

        self.features = nn.Sequential(
            nn.Conv2d(self.channels, self.channels_list[0], kernel_size=self.filter_size, padding=1),
            nn.BatchNorm2d(self.channels_list[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.get_blocks(self.channels_list[0], self.channels_list[1], self.num_filters_list[0], self.num_filters_list[1]),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Linear(self.num_filters_list[-1], self.num_classes)

    def get_blocks(self, in_channels, out_channels, num_filters1, num_filters2):
        layers = []
        for i in range(self.num_layers):
            layers.append(nn.Conv2d(in_channels, num_filters1, kernel_size=self.filter_size, padding=1))
            layers.append(nn.BatchNorm2d(num_filters1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = num_filters1

        layers.append(nn.Conv2d(num_filters1, num_filters2, kernel_size=self.filter_size, padding=1))
        layers.append(nn.BatchNorm2d(num_filters2))
        layers.append(nn.ReLU(inplace=True))

        return layers

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()