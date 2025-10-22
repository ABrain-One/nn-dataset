import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum']
        )
        self.clip_grad = 3                               

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
                               
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.dropout_p = prm['dropout']

                                                                 
        layers = []
                     
        layers.append(DoubleConv(self.in_channels, 64))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                      
        layers.append(DoubleConv(64, 128))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                     
        layers.append(DoubleConv(128, 256))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                      
        layers.append(DoubleConv(256, 512))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.features = nn.Sequential(*layers)
                                                   
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                                                   
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_p),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def supported_hyperparameters(self):
        return {'lr', 'momentum', 'dropout'}

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)