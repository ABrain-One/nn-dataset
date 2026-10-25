import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Net class with helper Block (if not exists already in previous models)
class Block(nn.Module):
    """Grouped convolution and batch normalization"""
    def __init__(self, in_channels, out_channels, stride, expansion):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Net(nn.Module):
    """ImageNet-style 5-stage block RS block (RS stands for Residual Scale)"""

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        # Hyperparameters can be accessed with prm: prm['lr'], prm['momentum']
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            Block(32, 32, 1, 1),
            Block(32, 32, 1, 1),
            Block(32, 32, 2, 4),

            Block(32, 32, 1, 1),
            Block(32, 32, 1, 1),
            Block(32, 32, 1, 1),
            Block(32, 32, 1, 1),
            Block(32, 32, 1, 1),
            Block(32, 32, 1, 1),
            Block(32, 32, 1, 1),

            Block(32, 32, 2, 4),
            Block(32, 32, 1, 1),
            Block(32, 32, 1, 1),

            nn.AdaptiveAvgPool2d((6, 6)),
        )

        self.classifier = nn.Linear(32 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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