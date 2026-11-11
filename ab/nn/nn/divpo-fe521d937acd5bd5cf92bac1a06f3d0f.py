import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # handle the channel order as per the guide
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.module1 = Block(32, 32)
        self.module2 = Block(32, 64)
        self.module3 = Block(64, 32)
        self.module4 = Block(32, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.module1(x)
        x = self.module2(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.module3(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.module4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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

# Optional: helper block below, if you need it in your design
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.layer(x)


def get_num_params(model):
    return sum(x.nelement() for x in model.parameters())


_ = Net(in_shape=[3, 32, 32], out_shape=[10], prm={}, device=torch.device('cpu'))  # Dummy to test for param count
print('Cell.params:', get_num_params(Block(32, 32)))
print('ensure < 500000:', get_num_params(Net(in_shape=[3, 32, 32], out_shape=[10], prm={}, device=torch.device('cpu'))) <= 500000)