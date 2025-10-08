import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr': 0.001, 'momentum': 0.9, 'dropout': 0.2}

class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4
        )
                                                  
        self.reducer = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.1
        )
                                                               
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
                               
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.reducer.step()
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        layers = []
        in_channels = in_shape[1]

                                   
        layers += [
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        in_channels = 64

                               
        layers += [
            nn.Conv2d(in_channels, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        in_channels = 128

                               
        layers += [
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        in_channels = 256

                               
        layers += [
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        in_channels = 512

                  
        self.features = nn.Sequential(*layers)
                                  
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
                    
        dropout = prm['dropout']
        classifier_input_features = in_channels * 6 * 6
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(classifier_input_features, 6144),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(6144, 4096),
            nn.GELU(),
            nn.Linear(4096, out_shape[0])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x