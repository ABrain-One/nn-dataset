import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape[0]
        self.hidden_dim = 640
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 512, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, self.hidden_dim)
        )
        
        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, batch_first=True),
            num_layers=6
        )
        
        # Final linear layer for output
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)
        
    def train_setup(self, dataset):
        if hasattr(dataset, 'word2idx'):
            self.word2idx = dataset.word2idx
            self.idx2word = dataset.idx2word
        
    def learn(self, images, captions=None, hidden_state=None):
        if captions is not None:
            if hasattr(captions, 'to'):
                captions = captions.to(self.device)
            else:
                captions = torch.tensor(captions, dtype=torch.long).to(self.device)
                
            if captions.ndim == 3:
                # [batch, num_captions, max_length]
                caps = captions[:,0,:] if captions.ndim == 3 else captions
                inputs = caps[:, :-1]
                targets = caps[:, 1:]
            else:
                inputs = captions[:, :-1]
                targets = captions[:, 1:]
                
            memory = self.encoder(images)
            logits = self.fc_out(self.decoder(inputs, memory))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            return loss
    
    def forward(self, images, captions=None, hidden_state=None):
        if captions is not None:
            if hasattr(captions, 'to'):
                captions = captions.to(self.device)
            else:
                captions = torch.tensor(captions, dtype=torch.long).to(self.device)
                
            if captions.ndim == 3:
                caps = captions[:,0,:] if captions.ndim == 3 else captions
                inputs = caps[:, :-1]
                targets = caps[:, 1:]
            else:
                inputs = captions[:, :-1]
                targets = captions[:, 1:]
                
            memory = self.encoder(images)
            logits = self.fc_out(self.decoder(inputs, memory))
            # Reshape logits to match targets
            logits = logits.permute(1, 0, 2)  # [T, B, V]
            logits = logits.contiguous().view(-1, self.vocab_size)
            targets = targets.contiguous().view(-1)
            
            # Calculate loss
            loss = F.cross_entropy(logits, targets, ignore_index=0)
            
            # Return hidden_state as the last token's features
            if logits.size(1) > 0:
                hidden_state = logits[:, -self.vocab_size:]
            else:
                hidden_state = torch.zeros(1, self.vocab_size).to(self.device)
                
            return logits, hidden_state, loss
        else:
            memory = self.encoder(images)
            return None, None, None