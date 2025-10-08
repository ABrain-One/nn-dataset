import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )
        if 'weight_decay' in prm:
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=prm['lr'],
                momentum=prm['momentum'],
                weight_decay=prm['weight_decay']
            )
        self._initialize_weights()

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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.hyperparameters = prm

                     
        layers = [
            nn.Conv2d(in_shape[1], 32, kernel_size=7, stride=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        in_channels = 32

                              
        layers += [
            nn.Conv2d(in_channels, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        in_channels = 128

        layers += [
            nn.Conv2d(in_channels, 440, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        in_channels = 440

        layers += [
            nn.Conv2d(in_channels, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        in_channels = 384

        layers += [
            nn.Conv2d(in_channels, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        in_channels = 192

                  
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

                    
        dropout_p = prm['dropout']
        classifier_input_features = in_channels * 6 * 6
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(classifier_input_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_shape[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x