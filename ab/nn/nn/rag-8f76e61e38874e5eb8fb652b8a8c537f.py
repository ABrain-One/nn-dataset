import torch
from torch import nn, optim
import torch.nn.functional as F
import os
import numpy as np

import torch, torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, features_dim, embed_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.features_dim = features_dim
        self.embed_dim = embed_dim
        self.scale = (self.features_dim // num_heads) ** -0.5

        self.to_q = nn.Linear(features_dim, features_dim, bias=False)
        self.to_kv = nn.Linear(embed_dim, features_dim * 2, bias=False)
        # self.to_out = nn.Linear(features_dim, features_dim)

    def forward(self, x, embedding):
        b, _, h, w = x.shape

        # Query from feature maps
        q = self.to_q(x.flatten(2).transpose(1, 2))  # Shape: (batch_size, height*width, features_dim)
        q = q.view(b, h * w, self.num_heads, self.features_dim // self.num_heads).permute(0, 2, 1, 3)

        # Key and value from embedding vector
        kv = self.to_kv(embedding.expand(b, -1)).view(b, 2, self.num_heads, self.features_dim // self.num_heads).permute(1, 2, 0, 3)
        k, v = kv[0], kv[1]

        # Scaled Dot-Product Attention
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)

        # Aggregate values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, h * w, self.features_dim)
        out = out.view(b, self.features_dim, h, w)  # Reshape to (batch_size, features_dim, height, width)
        # Final linear transformation
        # out = self.to_out(out)

        return out


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
        # layers += [ProvidedBlock(in_ch=32, out_ch=32)]
        # layers += [WinAttnAdapter(dim=32, window_size=(7,7), num_heads=4, block_cls=ProvidedBlock)]

        # Keep under parameter budget and end with a known channel count:
        self._last_channels = 32
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