import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# This is the output block that will be used in the final model
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)
        
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
        x = self.classifier(x)
        
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )
    
    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def prm_range(start, end, step):
    cur = start
    while cur < end:
        yield cur
        cur += step

def best_hyperparameters(train_data):
    best_acc = 0.0
    best_prm = None
    for lr in prm_range(0.01, 0.0001, -0.001):
        for momentum in prm_range(0.9, 0.999, -0.001):
            net = Net(in_shape=(3, 32, 32), out_shape=(10,), prm={'lr': lr, 'momentum': momentum}, device=torch.device('cpu'))
            net.learn(train_data)
            cur_acc = net.eval().criteria[0].cpu().item()
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_prm = {'lr': lr, 'momentum': momentum}
    return best_prm