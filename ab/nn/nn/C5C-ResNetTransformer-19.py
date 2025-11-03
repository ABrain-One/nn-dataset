import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :]
        x = x + pe
        return self.dropout(x)

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape
        self.in_channels = in_shape[1]
        self.hidden_dim = 768  # Ensure H>=640

        # Encoder CNN
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj_encoder = nn.Linear(1024, self.hidden_dim)

        # Decoder Transformer
        self.dec_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos_encoder = PositionalEncoding(self.hidden_dim)
        transformer_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, batch_first=True)
        self.transformer_dec = nn.TransformerDecoder(transformer_layer, num_layers=6)
        self.proj_decoder = nn.Linear(self.hidden_dim, self.vocab_size)

    def train_setup(self, optimizer: torch.optim, lr: float, momentum: float):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            param_group['momentum'] = momentum

    def learn(self, train_data):
        images, captions = train_data
        images = images.to(self.device, dtype=torch.float32)
        captions = captions.to(self.device, dtype=torch.long)
        
        memory = self.encoder(images)
        memory = self.proj_encoder(memory.flatten(1))
        memory = memory.unsqueeze(1)  # [B, 1, 768]
        
        tgt = captions[:, :-1]  # [B, T-1]
        embedded = self.dec_embedding(tgt)
        embedded = self.pos_encoder(embedded)
        
        mask = torch.triu(torch.ones(tgt.shape[-1], tgt.shape[-1]), diagonal=1).to(self.device)
        mask = torch.masked_fill(mask, mask == 1, -torch.inf)
        
        out = self.transformer_dec(embedded, memory, tgt_mask=mask)
        logits = self.proj_decoder(out)
        
        targets = captions[:, 1:]  # [B, T-1]
        loss = F.cross_entropy(logits, targets, ignore_index=0)
        
        return loss

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state: Optional[torch.Tensor] = None):
        memory = self.encoder(images)
        memory = self.proj_encoder(memory.flatten(1))
        memory = memory.unsqueeze(1)  # [B, 1, 768]
        
        if captions is not None:
            tgt = captions[:, :-1]  # [B, T-1]
            embedded = self.dec_embedding(tgt)
            embedded = self.pos_encoder(embedded)
            
            mask = torch.triu(torch.ones(tgt.shape[-1], tgt.shape[-1]), diagonal=1).to(self.device)
            mask = torch.masked_fill(mask, mask == 1, -torch.inf)
            
            out = self.transformer_dec(embedded, memory, tgt_mask=mask)
            logits = self.proj_decoder(out)
            
            # Return logits and None for hidden_state (Transformer doesn't have sequential hidden_state)
            return logits, None
        
        raise NotImplementedError("Inference mode not implemented in this version")

def supported_hyperparameters():
    return {'lr', 'momentum'}