import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.dropout = prm['dropout']

                                                            
        self.features = nn.Sequential(
                                 
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
                                 
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
                                 
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
                                 
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
                                 
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
                                 
            nn.Conv2d(96, 160, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
                                 
            nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
                                 
            nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
        )

                                  
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

                    
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(320 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(2048, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=0.0,
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()