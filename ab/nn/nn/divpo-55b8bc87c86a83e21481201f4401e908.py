import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = 32
        self.conv1 = nn.Conv2d(self.in_channels, self.features, 3, padding=1)
        self.prelu1 = nn.PReLU(self.features)
        self.conv2 = nn.Conv2d(self.features, self.features, 3, padding=1)
        self.prelu2 = nn.PReLU(self.features)
        self.conv3 = nn.Conv2d(self.features, self.features * 2, 3, stride=2, padding=1)
        self.prelu3 = nn.PReLU(self.features * 2)
        self.conv4 = nn.Conv2d(self.features * 2, self.features * 2, 3, padding=1)
        self.prelu4 = nn.PReLU(self.features * 2)
        self.conv5 = nn.Conv2d(self.features * 2, self.features * 2, 3, padding=1)
        self.prelu5 = nn.PReLU(self.features * 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(self.features * 2 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = self.conv5(x)
        x = self.prelu5(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum,
                                          weight_decay=1e-4)
        
    def learn(self, train_data):
        self.train()
        for data, target in train_data:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()