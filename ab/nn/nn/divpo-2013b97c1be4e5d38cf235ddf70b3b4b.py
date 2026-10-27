import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]  # Follow the new channel-first format
        self.num_classes = out_shape[0]
        
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.layer1 = Block(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = Block(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer3 = Block(128, 256, kernel_size=3, stride=2, padding=1)
        self.layer4 = Block(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Adaptive average pooling to ensure output is 4x4
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Linear(512 * 4 * 4, self.num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
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
            momentum=prm.get('momentum', 0.9)
        )
    
    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def __repr__(self):
        return f'Net(in_channels={self.in_channels}, num_classes={self.num_classes})'

def learn(model, train_data, prm):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train_setup(prm)
    model.learn(train_data)
    return model

def train_setup(model, prm):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train_setup(prm)
    return model.to(device)

def get_model(in_shape, out_shape, prm):
    model = Net(in_shape, out_shape, prm, device=torch.device('cpu'))
    return model