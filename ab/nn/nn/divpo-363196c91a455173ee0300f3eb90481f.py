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
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # derive from in_shape
        self.num_classes = out_shape[0]
        self.features = nn.Sequential(
            Block(self.in_channels, 32, stride=2),
            Block(32, 64, stride=2),
            nn.MaxPool2d(2, 2),
            Block(64, 128, stride=2),
            Block(128, 256, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Linear(256 * 6 * 6, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __repr__(self):
        return f"Net(in_channels={self.in_channels}, num_classes={self.num_classes})"

def learn(model, train_data, prm):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train_setup(prm)
    model.to(device)
    optimizer = model.optimizer
    criteria = model.criteria

    model.learn(train_data)

    # Evaluate the model on the training data
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Train Accuracy: {accuracy:.2f}%')
    return model

def train(prm, train_data):
    model = Net(in_shape=(3, 32, 32), out_shape=(10,), prm=prm, device=torch.device('cpu'))
    model = learn(model, train_data, prm)
    return model

def create_model(in_shape, out_shape, prm):
    model = Net(in_shape, out_shape, prm, device=torch.device('cpu'))
    return model