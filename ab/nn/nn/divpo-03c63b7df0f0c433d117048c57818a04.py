import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # should be 3 for RGB
        num_classes = out_shape[0]
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer1 = self.make_layer(64, 64, 3, 1)
        self.layer2 = self.make_layer(64, 128, 3, 2)
        self.layer3 = self.make_layer(128, 256, 3, 2)
        self.layer4 = self.make_layer(256, 512, 3, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, num_classes)
        
    def make_layer(self, inplanes, planes, blocks, stride):
        layers = [
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks):
            layers.append(
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()


# This block is for hyper-parameters tuning
class MyOptimizer(object):
    def __init__(self, prm, net):
        self.prm = prm
        self.net = net
        self.params = net.parameters()

    def __call__(self, epoch, learning_rate):
        if epoch == 0:
            self.optimizer = torch.optim.SGD(self.params,
                                            lr=learning_rate,
                                            momentum=self.prm.get('momentum', 0.9))
        return self.optimizer

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

# This block is for hyper-parameters tuning
class MyScheduler(object):
    def __init__(self, prm, optimizer):
        self.prm = prm
        self.optimizer = optimizer
        self.lr = None

    def step(self, epoch):
        if epoch == 0:
            self.lr = self.prm.get('lr', 0.01)
        else:
            self.lr = self.lr * self.prm.get('decay_rate', 0.1)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def get_lr(self):
        return self.lr

    def state_dict(self):
        return {'lr': self.lr}

    def load_state_dict(self, state_dict):
        self.lr = state_dict['lr']
        self.step(0)