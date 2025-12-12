import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.models import densenet

                                                 
class Net(densenet.DenseNet):
    
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__(num_classes=out_shape[0])
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.criteria = nn.CrossEntropyLoss()
        self.class_number = out_shape[0]
        
    def forward(self, x):
        return super().forward(x)
    
    def train_setup(self, prm):
        self.to(self.device)
                                                                                                                           
                                                                      
                                         
                                        
        
                                                        
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=prm['lr'], momentum=prm['momentum'])
                                                    
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
            
    def learn(self, train_loader):
        self.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  

def supported_hyperparameters():
    return {'lr': 0.01, 'momentum': 0.9}