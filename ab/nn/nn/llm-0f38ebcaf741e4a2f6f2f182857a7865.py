import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.n_classes = out_shape[0] if out_shape else 10
        self.dropout_p = 0.08362963662685266
        self.norm_layer = nn.InstanceNorm2d
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.init_features = 32
        self.layer1 = nn.Sequential(nn.Conv2d(in_shape[1], self.init_features, kernel_size=3, stride=1, padding=1, bias=False), self.norm_layer(self.init_features), self.activation, nn.Conv2d(self.init_features, 32, kernel_size=3, stride=1, padding=1, bias=False), self.norm_layer(32), self.activation, self.pool)
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False), self.norm_layer(64), self.activation, nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), self.norm_layer(64), self.activation, self.pool)
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False), self.norm_layer(128), self.activation, nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False), self.norm_layer(128), self.activation, self.pool)
        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False), self.norm_layer(256), self.activation, nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False), self.norm_layer(256), self.activation, self.pool)
        self.layer5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False), self.norm_layer(512), self.activation, nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False), self.norm_layer(512), self.activation, self.pool)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(self.dropout_p)
        self.fc = nn.Linear(512, self.n_classes)
        self.to(device)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
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