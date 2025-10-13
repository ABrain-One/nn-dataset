import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
                          
        self.in_shape = in_shape
        self.out_shape = out_shape
                            
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=4, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=prm['dropout'])
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
                                
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, out_shape[0])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        
        x = self.conv5(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
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
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()