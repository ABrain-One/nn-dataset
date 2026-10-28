import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        c_in = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        self.features = nn.Sequential()
        self.features.add_module('conv1', nn.Conv2d(c_in, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('norm1', nn.BatchNorm2d(64))
        self.features.add_module('act1', nn.ReLU(inplace=True))
        self.features.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module('conv2', nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False))
        self.features.add_module('norm2', nn.BatchNorm2d(128))
        self.features.add_module('act2', nn.ReLU(inplace=True))
        self.features.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module('conv3', nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False))
        self.features.add_module('norm3', nn.BatchNorm2d(256))
        self.features.add_module('act3', nn.ReLU(inplace=True))
        self.features.add_module('pool3', nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module('conv4', nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False))
        self.features.add_module('norm4', nn.BatchNorm2d(512))
        self.features.add_module('act4', nn.ReLU(inplace=True))
        self.features.add_module('pool4', nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential()
        self.classifier.add_module('dropout', nn.Dropout(p=0.3))
        self.classifier.add_module('fc', nn.Linear(512, n_classes))

    def forward(self, x):
        x = self.features(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()