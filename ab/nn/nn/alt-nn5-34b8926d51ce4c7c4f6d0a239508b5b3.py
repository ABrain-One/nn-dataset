import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        
                                                  
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        
                                                        
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=3, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout2d(p=0.5)
        
        self.conv4 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
                             
        self.attention = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=1)
        )
        
                    
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm['dropout']),
            nn.Linear(384 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm['dropout']),
            nn.Linear(4096, self.num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)
        
        x = self.relu2(self.conv2(x))
        x = self.maxpool2(x)
        
        x = self.relu3(self.conv3(x))
        x = self.dropout3(x)
        
        x = self.relu4(self.conv4(x))
        x = self.maxpool3(x)
        
                         
        x = x * F.softmax(self.attention(x), dim=1)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()