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
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # handle in_shape format
        self.num_classes = int(out_shape[0])
        self.features = nn.Sequential(
            Block(self.in_channels, 64),
            nn.MaxPool2d(2, 2),
            Block(64, 128),
            nn.MaxPool2d(2, 2),
            Block(128, 256),
            nn.MaxPool2d(2, 2),
            Block(256, 512),
            nn.MaxPool2d(2, 2),
            Block(512, 512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

    def __repr__(self):
        return f"Net(in_channels={self.in_channels}, num_classes={self.num_classes})"

def learn(model, train_data, prm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train_setup(prm)
    model.to(device)
    optimizer = model.optimizer
    criterion = model.criteria[0]

    model.train()
    for inputs, labels in train_data:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def train_setup(model, prm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train_setup(prm)
    model.to(device)

def get_model(in_shape, out_shape, prm):
    model = Net(in_shape, out_shape, prm, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count <= 500000, f"Total parameter count ({param_count}) exceeds limit by {param_count - 500000}."
    return model