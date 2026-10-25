import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks BEFORE Net
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expansion=4):
        super(Block, self).__init__()
        hidden_dim = int(out_channels * expansion)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.ReLU6(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.act2 = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_res_connect:
            out = out + x

        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layers = nn.Sequential(
            Block(32, 32, kernel_size=3, stride=1, expansion=4),
            Block(32, 32, kernel_size=3, stride=1, expansion=4),
            Block(32, 32, kernel_size=3, stride=1, expansion=4),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
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
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()