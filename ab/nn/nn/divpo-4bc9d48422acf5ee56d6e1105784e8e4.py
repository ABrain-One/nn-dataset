import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Follow the guide: use in_shape[1] for channels
        num_classes = out_shape[0]
        self.conv1 = Block(in_channels, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Block(64, 128, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Block(128, 256, stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def parameters(self):
        return super().parameters()

    def __to_device(self, data, device):
        if isinstance(data, (list, tuple)):
            return type(data)(self.__to_device(d, device) for d in data)
        return data.to(device)

    def get_parameter_groups(self, lr, weight_decay):
        # Group parameters into those that will experience weight decay and those that won't
        no_decay = {'bias', 'LayerNorm.bias', 'LayerNorm.weight'}
        lr_group_1, lr_group_2 = [], []
        for mn, mp in self.named_parameters():
            if any(nd in mn for nd in no_decay):
                lr_group_1.append(mp)
            else:
                lr_group_2.append(mp)
        return [{'params': lr_group_1, 'weight_decay': weight_decay, 'lr': lr},
                {'params': lr_group_2, 'weight_decay': weight_decay, 'lr': lr}]