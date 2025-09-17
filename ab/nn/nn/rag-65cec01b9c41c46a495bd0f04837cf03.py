import math
import torch
import torch.nn as nn

import torch, torch.nn as nn



class Conv(nn.Module):
    def __init__(self, inp, oup, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(inp, oup, k, s, self._pad(k, p), d, g, False)
        self.norm = nn.BatchNorm2d(oup)
        self.act = nn.SiLU(inplace=True) if act is True else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    @staticmethod
    def _pad(k, p=None):
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p




class Residual(nn.Module):
    def __init__(self, inp, g=1, k=(3, 3), e=0.5):
        super().__init__()
        self.conv1 = Conv(inp, int(inp * e), k[0], 1)
        self.conv2 = Conv(int(inp * e), inp, k[1], 1, g=g)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))




class CSPBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv(2 * (out_ch // 2), out_ch)
        self.res_m = torch.nn.Sequential(Residual(out_ch // 2, e=1.0),
                                         Residual(out_ch // 2, e=1.0))

    def forward(self, x):
        y = self.res_m(self.conv1(x))
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))


import torch.nn.functional as F

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

        # Insert the provided block here; repeat/adapt as needed.
        # If the block expects (B*, N, C) tokens, create a WinAttnAdapter(dim=32, window_size=(7,7), num_heads=4, block_cls=ProvidedBlock, ...)
        # If the block is 2D (NCHW->NCHW), just append it after the stem.

        # Example patterns you may use (choose ONE appropriate to the block):
        layers += [CSPBlock(32, 64)]

        # Keep under parameter budget and end with a known channel count:
        self._last_channels = 64
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