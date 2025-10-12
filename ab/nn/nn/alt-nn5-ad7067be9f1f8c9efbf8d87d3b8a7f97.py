import torch
import torch.nn as nn
import torch.optim as optim

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.dropout_rate = prm['dropout']

                            
        self.conv1 = nn.Conv2d(in_shape[1], 96, kernel_size=7, stride=3, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten())

                    
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_shape[0])
        )

                        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)

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
        x = self.maxpool4(x)

        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        self.to(self.device)
        learning_rate = float(prm['lr'])
        momentum = float(prm['momentum'])
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=5e-4
        )
        self.to(self.device)

    def learn(self, train_data: torch.utils.data.DataLoader) -> None:
        self.train()
        for batch_idx, (inputs, targets) in enumerate(train_data):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()