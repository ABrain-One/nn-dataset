import torch, torch.nn as nn


class ConditionalLayerNorm(nn.Module):
    def __init__(self, in_channels, cond_channels, eps=1e-5):
\
\
\
\
\
\
\
           
        super(ConditionalLayerNorm, self).__init__()
        self.eps = eps
        self.in_channels = in_channels
        self.cond_channels = cond_channels

        self.weight_transform = nn.Linear(cond_channels, in_channels)
        self.bias_transform = nn.Linear(cond_channels, in_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight_transform.weight, 0.0)
        nn.init.constant_(self.weight_transform.bias, 1.0)
        nn.init.constant_(self.bias_transform.weight, 0.0)
        nn.init.constant_(self.bias_transform.bias, 0.0)

    def forward(self, x, c):
\
\
\
\
\
\
\
           
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)

        x_normalized = (x - mean) / (std + self.eps)

        gamma = self.weight_transform(c)
        beta = self.bias_transform(c)

        out = gamma * x_normalized + beta

        return out



import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
                                                          
        layers += [
            nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        ]

                                                                             
                                                        
                                                                                                     

                                                                         
        self._last_channels = 16
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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