import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def supported_hyperparameters():
    return {'lr','momentum'}


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        
    def forward(self, x):
        return self.conv(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0:64] = torch.sin(position / 1000.0**(4.0/(64.0)))
        pe[:, 0, 1:65] = torch.cos(position / 1000.0**(4.0/(64.0)))
        pe = pe[:, 0]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x is expected to be (batch, seq, d_model)
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        
        # Determine input channels and desired vocabulary size
        input_channels = int(in_shape[1])
        self.vocab_size = int(out_shape[0])
        
        # Set hidden dimension to 768 which is >=640
        self.hidden_dim = 768
        
        # Encoder section
        self.encoder = nn.ModuleList()
        # Input stem: Downsample from potentially variable input size to fixed 28x28
        self.encoder.append(BasicConv2d(input_channels, 32, kernel_size=7, stride=2, padding=3))
        self.encoder.append(nn.MaxPool2d(kernel_size=3, stride=2))
        
        # Increase channels gradually
        self.encoder.append(BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1))
        self.encoder.append(BasicConv2d(64, 64, kernel_size=3, stride=2, padding=1))
        self.encoder.append(BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1))
        
        # Prepare final feature extraction
        self.encoder.append(BasicConv2d(128, 128, kernel_size=3, stride=2, padding=1))
        self.encoder.append(BasicConv2d(128, 128, kernel_size=3, stride=2, padding=1))
        self.encoder.append(BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1))
        self.encoder.append(BasicConv2d(256, 256, kernel_size=3, stride=2, padding=1))
        self.encoder.append(BasicConv2d(256, 512, kernel_size=3, stride=2, padding=1))
        
        # Projection to hidden dimension
        self.encoder.append(nn.Flatten())
        self.encoder.append(nn.Linear(512 * 2 * 2, self.hidden_dim))
        
        # Decoder section - Transformer based
        # Text embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos_encoding = PositionalEncoding(self.hidden_dim)
        
        # Transformer decoder setup
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim, 
            nhead=8,
            dim_feedforward=2*self.hidden_dim,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.projection = nn.Linear(self.hidden_dim, self.vocab_size)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999))

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

    def forward(self, images, captions):
        # Process the images through the encoder
        x = images
        for layer in self.encoder:
            x = layer(x)
        
        # Project the image features to the hidden dimension
        x = x.reshape(-1, self.hidden_dim)
        
        # Embed the captions and add positional encoding
        y = self.embedding(captions)
        y = self.pos_encoding(y)
        
        # Run the transformer decoder
        out = self.transformer_decoder(x.unsqueeze(1), y)
        
        # Project the output to the vocabulary size
        logits = self.projection(out.reshape(-1, out.size(-1)))
        
        return logits