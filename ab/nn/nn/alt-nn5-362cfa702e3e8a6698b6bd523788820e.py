import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        in_channels, out_channels = in_shape[1], out_shape[0]
        
                                                        
                                                        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=3, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
                                                   
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
                                   
        self.inception_block = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
                                      
        self.pyramid_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=5),
            nn.Conv2d(192, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=1)
        )
        
                            
        self.fc1 = nn.Linear(256, 2048)
        self.dropout = nn.Dropout(prm['dropout'])
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, out_shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        
        x = self.inception_block(x)
        x = self.pyramid_pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data: torch.utils.data.DataLoader):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()