import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.lr = prm['lr']
        self.momentum = prm['momentum']
        self.features = 32
        self.gen_features = 32
        self.body = nn.Sequential(
            nn.Conv2d(self.in_channels, self.gen_features, 3, 1, 1),
            nn.BatchNorm2d(self.gen_features),
            nn.ReLU(True),
            self.make_layer(32, 32),
            self.make_layer(32, 32),
            self.make_layer(32, 32),
        )
        self.classifier = nn.Linear(self.gen_features, self.num_classes)
        
    def make_layer(self, inpt:int ,ch:int):
        layer = nn.Sequential(
            nn.Conv2d(inpt,ch,3,dilation=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch,ch, kernel_size=3,dilation=1,padding=1,bias=False),
            nn.LeakyReLU(0.1,inplace=True) 
        )
        return layer
        
    def forward(self, x):
        x = self.body(x)
        avg_x = torch.mean(x, 3)
        avg_x = torch.mean(avg_x, 2)
        return self.classifier(avg_x)
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )
        
    def learn(self, train_data):
        self.train()
        for data, target in train_data:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criteria(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
            self.optimizer.step()