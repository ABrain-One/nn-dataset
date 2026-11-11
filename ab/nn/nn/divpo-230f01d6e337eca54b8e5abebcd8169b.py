import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pre_act = nn.Sequential(nn.AdaptiveAvgPool2d(6),
                                     nn.Conv2d(32, 32, 1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))
        self.drop = nn.Dropout(p=prm.get('dropout', 0.2))
        self.classifier = nn.Linear(32 * 6 * 6, out_shape[0])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pre_act(x)
        x = self.drop(x)
        B = x.shape[0]
        x = x.view(B, -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
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