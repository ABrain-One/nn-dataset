import torch
import torch.nn as nn
import torch.nn.functional as F

import torch, torch.nn as nn



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)




class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



def supported_hyperparameters():
 return {'lr','momentum'}

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
  # Stable 2D stem to avoid channel/shape mismatches
  layers += [
   nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
   nn.BatchNorm2d(32),
   nn.ReLU(inplace=True),
  ]

  # Using the provided block
  layers += [DoubleConv(32, 64)]
  layers += [Down(64, 128)]

  # Keep under parameter budget and end with a known channel count:
  self._last_channels = 128
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