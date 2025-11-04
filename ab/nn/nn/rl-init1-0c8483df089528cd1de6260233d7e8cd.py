import torch
from torch import nn, optim
from torchvision.models import vgg11_bn

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.criteria = nn.CrossEntropyLoss()
        self.model = vgg11_bn(num_classes=out_shape[0])
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.dropout_value = prm['dropout']

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_loader):
        self.train()
        running_loss = 0
        corrects = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = self.criteria(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == targets.data)
            total += inputs.size(0)
            loss.backward()
            self.optimizer.step()
        epoch_acc = corrects.double() / total
        print('Train Epoch Accuracy: {}/{} ({:.0f}%)\n'.format(corrects, total, 100. * epoch_acc))

def supported_hyperparameters():
    return ['lr', 'momentum', 'dropout']