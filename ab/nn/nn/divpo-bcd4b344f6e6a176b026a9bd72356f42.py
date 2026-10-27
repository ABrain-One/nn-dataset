import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Should be 3 for RGB images
        num_classes = int(out_shape[0])
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.batchnorm5 = nn.BatchNorm1d(4096)
        self.relu5 = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.5))
        
        self.fc2 = nn.Linear(4096, 4096)
        self.batchnorm6 = nn.BatchNorm1d(4096)
        self.relu6 = nn.ReLU(inplace=True)
        
        self.classifier = nn.Linear(4096, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        
        x = self.maxpool1(x)
        
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        
        x = self.maxpool2(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batchnorm6(x)
        x = self.relu6(x)
        
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()