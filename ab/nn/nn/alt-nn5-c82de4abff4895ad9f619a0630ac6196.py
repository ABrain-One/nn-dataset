import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
                                                                    
        input_channels = in_shape[1]
        
                                                      
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
                                         
            nn.Conv2d(32, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
                                                              
            nn.Conv2d(192, 384, kernel_size=3, padding=1, groups=32, bias=False),
            nn.ReLU(inplace=True),
            
                                         
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=32, bias=False),
            nn.ReLU(inplace=True),
            
                                      
            nn.Conv2d(384, 192, kernel_size=3, padding=1, groups=32, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=prm['dropout']),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
                                  
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
                                 
        classifier_input_features = 192 * 6 * 6
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm['dropout']),
            nn.Linear(classifier_input_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm['dropout']),
            nn.Linear(4096, out_shape[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            lr=prm['lr'],
            momentum=prm['momentum'],
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()