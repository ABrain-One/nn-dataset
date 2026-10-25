import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.5))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, out_shape[0])
        self.to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_flops(self, x):
        flops = 0
        x = torch.rand_like(x)
        x = self.conv1(x)
        flops += x.numel() * (32 * 3 * 3 * 3 + 32)
        x = self.batchnorm1(x)
        flops += x.numel() * (32)
        x = self.relu(x)
        flops += x.numel() * (32)
        x = self.maxpool(x)
        x = self.conv2(x)
        flops += x.numel() * (64 * 32 * 3 * 3 + 64)
        x = self.batchnorm2(x)
        flops += x.numel() * (64)
        x = self.relu(x)
        flops += x.numel() * (64)
        x = self.maxpool(x)
        x = self.avgpool(x)
        flops += x.numel() * (64 * 6 * 6)
        flops += self.classifier.weight.numel() * (6 * 6 * 6)
        return flops