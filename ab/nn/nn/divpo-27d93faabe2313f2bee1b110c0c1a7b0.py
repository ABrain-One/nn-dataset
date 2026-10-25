import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = Block(64, 128, stride=2)
        self.block3 = Block(128, 128)
        self.block4 = Block(128, 256, stride=2)
        self.block5 = Block(256, 256)
        self.block6 = Block(256, 256)
        self.block7 = Block(256, 512, stride=2)
        self.block8 = Block(512, 512)
        self.block9 = Block(512, 512)
        self.block10 = Block(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# The following code is used for testing the model and should not be modified.

def test_net(net_class, in_shape, out_shape, prm):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net_class(in_shape, out_shape, prm, device)
    assert isinstance(net, Net)
    assert net.in_channels == in_shape[1]
    assert net.num_classes == out_shape[0]
    assert net.device == device

    inputs = torch.randn(2, in_shape[1], 32, 32)
    outputs = net(inputs)
    assert outputs.shape[0] == 2 and outputs.shape[1] == out_shape[0]

    prm = {'lr': 0.01, 'momentum': 0.9}
    net.train_setup(prm)
    assert net.optimizer is not None
    assert net.criteria[0] is not None

    train_data = [(inputs, torch.randn(2, out_shape[0])) for _ in range(2)]
    net.learn(train_data)
    assert net.conv1 is not None
    assert net.block1 is not None
    assert net.pool1 is not None
    assert net.block2 is not None
    assert net.block3 is not None
    assert net.block4 is not None
    assert net.block5 is not None
    assert net.block6 is not None
    assert net.block7 is not None
    assert net.block8 is not None
    assert net.block9 is not None
    assert net.block10 is not None
    assert net.avgpool is not None
    assert net.classifier is not None

    print("All tests passed!")

if __name__ == '__main__':
    in_shape = (3, 32, 32)
    out_shape = (10,)
    prm = {'lr': 0.01, 'momentum': 0.9}
    test_net(Net, in_shape, out_shape, prm)