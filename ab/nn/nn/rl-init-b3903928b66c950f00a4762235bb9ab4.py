import math
from typing import Tuple
import torch
from torch import nn
from torch.nn import (
    ModuleDict,
    ModuleList,
    Linear,
    Module,
    Parameter,
    GRU,
    Dropout,
    LayerNorm,
    Embedding,
    GELU
)
from torch.nn import functional

import torch, torch.nn as nn



class AttentionTalkingHead(Module):
                                                                                                                
                                                                                                       
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_l = Linear(num_heads, num_heads)
        self.proj_w = Linear(num_heads, num_heads)
        self.proj_drop = Dropout(proj_drop)

    def forward(self, x):
        batch, max_atoms, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, max_atoms, 3, self.num_heads, dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch, max_atoms, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class FFN(Module):
                                                                           

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class LayerScaleBlock(Module):
                                                                                                                
                                                 
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            act_layer=nn.GELU,
            init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = AttentionTalkingHead(
            dim, num_heads, qkv_bias, qk_scale, attn_drop, drop
        )
        self.norm2 = LayerNorm(dim)
        self.ffn = FFN(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.gamma_1 = Parameter(init_values * torch.ones((dim,)), requires_grad=True)
        self.gamma_2 = Parameter(init_values * torch.ones((dim,)), requires_grad=True)

    def forward(self, x):
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        x = x + self.gamma_2 * self.ffn(self.norm2(x))
        return x


import torch.nn as nn
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
                                                          
        layers += [
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]

                                                                             
                                                        
                                                                                                     

                                                                         
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