import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.num_classes = out_shape[0]
        self.features = nn.Sequential(nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.5))
        self.classifier = nn.Linear(1024, self.num_classes)
        self.to(device)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.train()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()