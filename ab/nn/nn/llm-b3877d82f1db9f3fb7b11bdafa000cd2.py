import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

def _get_backbone(in_shape, out_shape, prm, device):
    conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False)
    bn1 = nn.BatchNorm2d(64)
    relu1 = nn.ReLU(inplace=True)
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
    bn2 = nn.BatchNorm2d(128)
    relu2 = nn.ReLU(inplace=True)
    conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
    bn3 = nn.BatchNorm2d(256)
    relu3 = nn.ReLU(inplace=True)
    conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
    bn4 = nn.BatchNorm2d(512)
    relu4 = nn.ReLU(inplace=True)
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    dropout = nn.Dropout(p=prm['dropout'])
    fc = nn.Linear(512, out_shape[0])
    return nn.Sequential(conv1, bn1, relu1, maxpool, conv2, bn2, relu2, conv3, bn3, relu3, conv4, bn4, relu4, avgpool, dropout, nn.Flatten(), fc).to(device)

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.learning_rate = float(prm['lr'])
        self.momentum_value = float(prm['momentum'])
        self.num_epochs = int(prm['epoch'])
        self._transform_name = prm['transform']
        self.batch = int(prm['batch'])
        self.dropout = float(prm['dropout'])
        self.backbone = _get_backbone(in_shape, out_shape, prm, device)
        self.to(device)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=float(prm['lr']), momentum=float(prm['momentum']))

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

    def forward(self, x):
        return self.backbone(x)