

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape)
        
        # Encoder configuration
        self.in_channels = int(in_shape[1])
        self.h, self.w = in_shape[2], in_shape[3]
        self.patch_size = 4
        
        # Encoder backbone with increasing capacity
        self.stem = nn.Conv2d(self.in_channels, 64, kernel_size=self.patch_size, stride=self.patch_size)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.project = nn.Linear(1024, self.vocab_size)
        self.dropout = nn.Dropout(0.2)
        
        # Decoder configuration
        self.hidden_dim = 768  # ≥640
        self.num_layers = 2
        
        # Decoder architecture
        self.rnn = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_dim,
            padding_idx=0
        )
        
        # Target masking utilities
        self.tgt_mask = None
    
    def init_zero_hidden(self, batch: int, device: torch.device):
        return (torch.randn(self.num_layers, batch, self.hidden_dim).to(device), 
                torch.randn(self.num_layers, batch, self.hidden_dim).to(device))
    
    def train_setup(self, prm: dict):
        self.to(self.device)
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images, captions = images.to(self.device), captions.to(self.device)
            logits = None
            if hasattr(self, 'forward'):
                out = self.forward(images, captions)
                logits = out[0] if isinstance(out, tuple) else out
            if logits is None:
                continue
            tgt = (captions[:,0,:] if captions.ndim==3 else captions)[:,1:]
            loss = self.criteria[0](logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)