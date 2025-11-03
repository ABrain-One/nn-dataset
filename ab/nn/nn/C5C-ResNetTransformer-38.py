

import math
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
        in_channels = int(in_shape[1])
        
        # Encoder configuration
        self.hidden_dim = 768
        
        # Build encoder: CNN-based feature extractor
        self.enc_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
        # Project visual features to language features
        self.proj_enc = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Decoder configuration
        self.decoder_layers = 6
        self.decoder_heads = 8
        
        # Build decoder: Transformer decoder with multi-head attention
        self.dec_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos_encoding = PositionalEncoding(self.hidden_dim)
        self.dec_pos_encoding = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Decoder layers and final projection
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.decoder_heads,
            dim_feedforward=2*self.hidden_dim,
            batch_first=True,
            dropout=0.1 if 'dropout' in prm else 0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.decoder_layers)
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)
        
        self.to(device)

    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.zeros(batch, self.decoder_heads, self.hidden_dim, device=device)
    
    def proj_query_init(self):
        return torch.empty(0, self.hidden_dim, device=self.device)

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
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

        self.optimizer = torch.optim.Adam