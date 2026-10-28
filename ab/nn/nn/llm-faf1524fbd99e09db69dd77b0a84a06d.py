import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'batch', 'transform', 'lr', 'momentum', 'dropout', 'epoch'}

def transform(norm):
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(*norm)])

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.dropout = float(prm['dropout'])
        self.batch = int(prm['batch'])
        self.learning_rate = float(prm['lr'])
        self.momentum_value = float(prm['momentum'])
        self.num_epochs = int(prm['epoch'])
        self._transform_name = prm['transform']
        self.backbone = nn.Sequential(nn.Conv2d(in_shape[1], 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        self.head = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(512, out_shape[0]))
        self.to(device)

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.head(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()