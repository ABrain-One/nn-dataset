import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        if len(in_shape) == 4:  # handle in_shape as (batch, channels, height, width)
            self.channels, self.height, self.width = in_shape[1], in_shape[2], in_shape[3]
        else:
            self.channels = in_shape[0]
        self.num_classes = int(out_shape[0])
        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
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
            self.optimizer.step()

    def __str__(self):
        return self.__class__.__name__ + '(' + \
            'in_channels=' + str(self.channels) + ', ' + \
            'out_channels=' + str(128) + ', ' + \
            'kernel_size=3, ' + \
            'padding=1, ' + \
            'num_classes=' + str(self.num_classes) + ')'

    def __repr__(self):
        return self.__class__.__name__ + '(' + \
            'in_channels=' + str(self.channels) + ', ' + \
            'out_channels=' + str(128) + ', ' + \
            'kernel_size=3, ' + \
            'padding=1, ' + \
            'num_classes=' + str(self.num_classes) + ')'

def get_model(in_shape, out_shape, prm, device):
    model = Net(in_shape, out_shape, prm, device)
    return model, prm['lr'], prm['momentum']