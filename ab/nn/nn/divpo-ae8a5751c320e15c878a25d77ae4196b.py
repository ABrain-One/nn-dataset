import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, expansion=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * expansion, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels * expansion),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * expansion, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.conv(x) + self.shortcut(x))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.block1 = Block(32, 32, expansion=1)
        self.block2 = Block(32, 32, expansion=1)
        self.block3 = Block(32, 64, stride=2, expansion=1)
        self.block4 = Block(64, 64, expansion=1)
        self.block5 = Block(64, 128, stride=2, expansion=1)
        self.block6 = Block(128, 128, expansion=1)
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = nn.MaxPool2d(2, 2)(x)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.MaxPool2d(2, 2)(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool(x)
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

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}


# Calculate the number of parameters
def get_n_params(model_obj):
    pp = 0
    for p in list(model_obj.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


_ = Net(in_shape=(3, 32, 32), out_shape=(10,), prm={}, device=torch.device('cpu'))
prm = {'lr': 0.01, 'momentum': 0.9}
model = Net(in_shape=(3, 32, 32), out_shape=(10,), prm=prm, device=torch.device('cpu'))

# Get and print the total number of parameters
total_params = get_n_params(model)
print(total_params)
assert total_params <= 500000, 'Total parameter count must be <= 500000'