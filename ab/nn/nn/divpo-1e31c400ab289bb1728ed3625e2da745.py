import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Follow the guide to derive this from in_shape
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            Block(32, 64, stride=1),
            nn.MaxPool2d(2, 2),
            Block(64, 128, stride=1),
            nn.MaxPool2d(2, 2)
        )
        self.layer1 = Block(128, 256, stride=1)
        self.layer2 = nn.Sequential(
            Block(256, 256, stride=1),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            Block(256, 512, stride=1),
            nn.MaxPool2d(2, 2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

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
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __str__(self):
        return "EdgeFriendlyResNet32"

def learn(model, train_data, prm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train_setup(prm)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=prm['lr'], momentum=prm['momentum'])
    for inputs, labels in train_data:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return model

def train_setup(model, prm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train_setup(prm)
    model.to(device)
    return model.to(device), nn.CrossEntropyLoss().to(device), torch.optim.SGD(model.parameters(), lr=prm['lr'], momentum=prm['momentum'])

def get_model(in_shape, out_shape, prm):
    model = Net(in_shape, out_shape, prm, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count <= 500000, f"Model is too large ({param_count} params)"
    return model

def supported_hyperparameters():
    return {'lr', 'momentum'}