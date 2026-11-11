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

        self.features = self.image_size // 4
        self.conv1 = nn.Conv2d(self.in_channels, 6, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 36, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(36)

        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        ch = 36
        for i in range(3):
            self.convs.append(nn.Conv2d(ch, ch, 3, 1, 1))
            self.bns.append(nn.BatchNorm2d(ch))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(ch, self.num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = x ** 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = x ** 8
        for conv, bn in zip(self.convs, self.bns):
            y = conv(x)
            y = bn(y)
            y = y ** 2
            x = x + y

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (self.criterion,)
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm.get('lr', 0.01), 
            momentum=prm.get('momentum', 0.9)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()