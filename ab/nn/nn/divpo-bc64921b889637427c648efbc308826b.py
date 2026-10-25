import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        if len(in_shape) == 4:  # channels-first C x H x W
            in_channels = in_shape[1]  # should be 3 for CIFAR-10
        else:
            in_channels = in_shape[0]  # should be 3 for CIFAR-10
        num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bn2(self.conv2(x))
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bn3(self.conv3(x))
        x = nn.functional.relu(x)
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
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0001)
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

    def __repr__(self):
        return f"Net(in_shape={self.conv1.in_channels}, out_shape={self.classifier.out_features}, prm={len(self.optimizer.param_groups)}, device={self.device})"

def train_setup(model, prm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train_setup(prm)
    model.device = device

def learn(model, train_data):
    model.learn(train_data)
def supported_hyperparameters():
    return {'lr', 'momentum', 'weight_decay'}