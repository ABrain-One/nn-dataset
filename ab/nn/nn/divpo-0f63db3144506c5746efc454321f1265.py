import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(32, 32, 3)
        self.layer2 = self._make_layer(32, 64, 3, stride=2)
        self.layer3 = self._make_layer(64, 128, 3, stride=2)
        self.layer4 = self._make_layer(128, 256, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Linear(256 * 6 * 6, self.num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layer = []
        layer += [nn.Conv2d(in_channels, out_channels, kernel_size=3,
                            stride=stride, padding=1, bias=False)]
        layer += [nn.BatchNorm2d(out_channels)]
        layer += [nn.ReLU(inplace=True)]
        for i in range(1, blocks):
            layer += [nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)]
            layer += [nn.BatchNorm2d(out_channels)]
            layer += [nn.ReLU(inplace=True)]
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.features(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), 
                                          lr=self.learning_rate, 
                                          momentum=self.momentum, 
                                          weight_decay=1e-4)
        
    def learn(self, train_loader):
        self.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3.0)
            self.optimizer.step()