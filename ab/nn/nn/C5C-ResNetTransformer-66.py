

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
        self.vocab_size = int(out_shape[0]) if isinstance(out_shape, tuple) else int(out_shape)
        self.hidden_dim = 768
        
        # TODO: Replace self.encoder with custom encoder producing memory tensor [B, S, H] where H >= 640
        in_channels = int(in_shape[1])
        height, width = in_shape[2], in_shape[3]
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * ((height + 1)//4) * ((width + 1)//4), self.hidden_dim),
            nn.ReLU()
        )
        
        # TODO: Replace self.rnn with custom decoder implementing forward(inputs, hidden_state, features) -> (logits, hidden_state)
        self.decoder_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=min(8, self.hidden_dim//4))
        self.decoder_transformer = nn.TransformerDecoder(decoder_layer, num_layers=2, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.optimizer = None
    
    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.zeros(batch, self.hidden_dim, device=device)
    
    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = self.criterion.to(self.device)
    
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:, :, 0].long().to(self.device) if len(captions.shape) == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            memory = self.encoder(images)
            embedded_inputs = self.decoder_embedding(inputs)
            
            # Calculate relative sequence length for memory
           