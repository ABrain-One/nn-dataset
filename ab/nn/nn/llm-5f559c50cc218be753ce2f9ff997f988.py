import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        c_in = 3
        n_classes = out_shape[0] if out_shape else 10
        start_chan = 64
        self.entry = nn.Sequential(nn.Conv2d(c_in, start_chan, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(start_chan), nn.ReLU(inplace=True))
        self.block1 = nn.Sequential(nn.Conv2d(start_chan, start_chan * 2, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(start_chan * 2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(start_chan * 2, start_chan * 2, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(start_chan * 2), nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(nn.Conv2d(start_chan * 2, start_chan * 4, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(start_chan * 4), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(start_chan * 4, start_chan * 4, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(start_chan * 4), nn.ReLU(inplace=True))
        self.block3 = nn.Sequential(nn.Conv2d(start_chan * 4, start_chan * 8, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(start_chan * 8), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(start_chan * 8, start_chan * 8, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(start_chan * 8), nn.ReLU(inplace=True))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(start_chan * 8, n_classes)
        self.to(device)

    def forward(self, x):
        x = self.entry(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
        self.max_batches = prm.get('max_batches', None)
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
            if self.max_batches is not None and i >= self.max_batches:
                break
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()