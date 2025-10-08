import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), 
                                         lr=prm['lr'], 
                                         momentum=prm['momentum'],
                                         weight_decay=0.0001)                      
                                                           
        if 'params' in prm and 'params' != 'all':
            self.optimizer = torch.optim.SGD([{
                'params': prm['params'],
                'lr': prm['lr'] * 0.1,
                'momentum': prm['momentum'],
                'weight_decay': 0.0001
            }],
                                         lr=prm['lr'], 
                                         momentum=prm['momentum'],
                                         weight_decay=0.0001)

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

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.out_features = out_shape[0]

                                                     
        self.features = nn.Sequential(
                         
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
                          
            nn.Conv2d(64, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
                         
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
                          
            nn.Conv2d(384, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((6, 6))
        )

                               
        self.dropout = nn.Dropout2d(p=prm['dropout']) if 'dropout' in prm else nn.Dropout2d(p=0.2)

                                
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm['dropout']),
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(4096, self.out_features)
        )

                               
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _get_parameter_groups(self):
        groups = {'params': [], 'params_with_weight_decay': []}
        for name, param in self.named_parameters():
            if param.dim() >= 1 and 'weight' in name:
                groups['params_with_weight_decay'].append(param)
            elif 'bn' in name or 'bias' in name:
                groups['params'].append(param)
        return groups