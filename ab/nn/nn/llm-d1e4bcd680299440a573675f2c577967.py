import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.hornet_block = self.build_hornet_block()
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=False)]
        return nn.Sequential(*layers)

    def build_hornet_block(self):
        return nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32), nn.Conv2d(32, 32, kernel_size=1), nn.ReLU(inplace=False), nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32), nn.Conv2d(32, 32, kernel_size=1), nn.ReLU(inplace=False), nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32), nn.Conv2d(32, 32, kernel_size=1), nn.ReLU(inplace=False))

    def forward(self, x):
        x = self.features(x)
        x = self.hornet_block(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=0.0005)

    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = (data.to(self.device), target.to(self.device))
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()