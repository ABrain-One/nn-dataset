import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'batchnorm'}

class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=0.0001                                         
        )
                                                      
        self._apply_batchnorm()
        
    def _apply_batchnorm(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, 'batchnorm') and module.batchnorm is None:
                    bn = nn.BatchNorm2d(module.out_channels)
                    module.batchnorm = bn
                    module.register_buffer('running_mean', torch.zeros(module.out_channels))
                    module.register_buffer('running_var', torch.ones(module.out_channels))
                    module.register_buffer('eps', 1e-5)

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
        self.hyperparameters = prm
        
                     
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64)
        )
        
                                                                        
        self.features.add_module('conv2', nn.Conv2d(64, 128, kernel_size=5, padding=2))
        self.features.add_module('relu2', nn.ReLU(inplace=True))
        self.features.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module('bn2', nn.BatchNorm2d(128))
        
        self.features.add_module('conv3', nn.Conv2d(128, 256, kernel_size=3, padding=1))
        self.features.add_module('relu3', nn.ReLU(inplace=True))
        self.features.add_module('bn3', nn.BatchNorm2d(256))
        
        self.features.add_module('conv4', nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.features.add_module('relu4', nn.ReLU(inplace=True))
        self.features.add_module('pool3', nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module('bn4', nn.BatchNorm2d(384))
        
                    
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm['dropout']),
            nn.Linear(384 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm['dropout']),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_shape[0])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x