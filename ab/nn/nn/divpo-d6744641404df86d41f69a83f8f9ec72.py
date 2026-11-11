import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.conv1 = self._conv(3, 36, 1,  padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, 2, padding=0)
        self.conv2 = self._conv(36, 72, 1, padding=1, bias=False)
        self.adj_channel = 72
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.linear = self._linear(72*6*6, 10)

        self._initialize_weights()
        self.device = device
        self.prm = prm
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.0)) if prm.get('dropout', 0.0) > 0.0 else None


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(-1,self.adj_channel*6*6)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        return x

    def _conv(self, in_ch, out_ch, kernel_size, padding=0, stride=1, bias=False):
        return nn.Conv2d(in_ch, out_ch, kernel_size, 
                        stride=stride, padding=padding, bias=bias)

    def _linear(self, in_ch, out_ch, bias=True):
        return nn.Linear(in_ch, out_ch, bias=bias)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=200,eta_min=0.)

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
            self.scheduler.step()