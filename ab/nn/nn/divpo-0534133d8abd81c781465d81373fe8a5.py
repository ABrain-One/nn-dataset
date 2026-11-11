import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    # Block to generate 1x1 > 3x3 > 1x1 convs with 1/2/3/4-channel input & 32-channel output
    def __init__(self, in_chs, out_chs, path_chs):
        super().__init__()
        # 1x1 conv, use in_chs=32 if path_chs is None (skip)
        self.conv1 = nn.Conv2d(in_chs if path_chs is None else path_chs, out_chs, 1)
        # 3x3 conv
        self.conv2 = nn.Conv2d(out_chs, out_chs, 3, padding=1)
        # 1x1 conv
        self.conv3 = nn.Conv2d(out_chs, out_chs, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.blocks = Block(32, 32, None)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, out_shape[0])
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.5))
        self.train_setup(prm)

    def forward(self, x):
        # x: [0, 1] 32 x 32 RGB
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

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