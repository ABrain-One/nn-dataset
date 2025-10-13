import torch
import torch.nn as nn

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
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
                         
        self.res_block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=2, bias=False),
            nn.BatchNorm2d(64)
        )
        
        self.res_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=2, bias=False),
            nn.BatchNorm2d(64)
        )

                                  
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
                                 
        self.dropout = nn.Dropout(p=prm['dropout'])
        self.classifier = nn.Sequential(
            nn.Linear(6 * 6 * 64, 4096),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
                              
        x = x + self.res_block1(x)
        x = x + self.res_block2(x)
        
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