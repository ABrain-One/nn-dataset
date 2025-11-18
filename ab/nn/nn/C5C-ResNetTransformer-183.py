

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
        self.vocab_size = int(out_shape[0])  # Assuming out_shape is tuple containing just vocab size
        in_channels = int(in_shape[1])
        
        # Create encoder
        enc = nn.Sequential(
            # Stem
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Transition Pooling
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Stage 1
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Stage 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Stage 3
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Flatten and project
            nn.Flatten(),
            nn.Linear(128, 768)  # Hidden dimension ≥640 (using 768)
        )
        self.encoder = enc
        
        # Create decoder
        dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=768, nhead=8, dim_feedforward=2048, batch_first=True),
            num_layers=6
        )
        dec_embed = nn.Embedding(self.vocab_size, 768)
        dec_pos_enc = nn.Parameter(torch.zeros(1, 768, dtype=torch.float32))
        nn.init.uniform_(dec_pos_enc, -0.1, 0.1)
        self.decoder = nn.Sequential(
            dec_embed,
            dec_pos_enc,
            dec
        ).to(device)
        self.final_proj = nn.Linear(768, self.vocab_size)
        
        # Initialize parameters
        self.to(self.device)
        self.train_setup(prm)

    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.empty(0, device=device), torch.empty(0, device=device)

    def train_setup(self, prm: dict):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=max(float(prm.get('lr', 1e-3)), 3e-4))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.criterion.to(self.device)

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encoder(images)  # Shape: [B, S, 768]
            embedded = self.decoder(inputs).transpose(0, 1)  # Shape: [T, B, 768]
            
            logits = self.final_proj(embedded.transpose(0, 1))  # Shape: [T, B, VOCAB_SIZE]
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)  # Shape: [B, S, 768]
        
        if captions is not None:
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]  # Shape: [B, T]
            
            embedded = self.decoder(inputs)  # Shape: [B, T, 768]
            logits = self.final_proj(embedded.transpose(0, 1))  # Shape: [T, B, VOCAB_SIZE]
            
            return logits, memory
        else:
            raise NotImplementedError