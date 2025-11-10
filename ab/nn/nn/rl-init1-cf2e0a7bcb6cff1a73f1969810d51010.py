import torch
from torch import nn

def vgg_like():
                        
    return Net((2, 32, 32), (10, ), {'lr': 0.001, 'momentum': 0.9, 'batch': 64, 'transform': None, 'epoch': 10}, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def supported_hyperparameters():
    return {'lr': 0.001, 'momentum': 0.9, 'batch': 64, 'transform': None, 'epoch': 10}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device

        self.learning_rate = prm['lr'] if 'lr' in prm else 0.001
        self.momentum = prm['momentum'] if 'momentum' in prm else 0.9
        self.batch_size = prm['batch'] if 'batch' in prm else 64
        self.transform = prm['transform'] if 'transform' in prm else None
        self.num_classes = out_shape[0] if len(out_shape) > 0 else 10
        self.epochs = prm['epoch'] if 'epoch' in prm else 10

        self.criteria = nn.CrossEntropyLoss()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)

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