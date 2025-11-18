import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def supported_hyperparameters():
    return {'lr','momentum'}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.view(1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CNNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, image_size=224, num_layers=6, num_heads=8):
        super().__init__()
        # CNN Feature Extractor with Spatial Reduction
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Linear projection to hidden dimension
        self.proj = nn.Linear(512 * ((image_size//16)//2 * ((image_size//16)//2)), out_channels)

    def forward(self, images):
        # Pass through CNN layers
        x = self.cnn(images)
        
        # Global average pooling to get fixed-length feature vectors
        pooled = F.adaptive_avg_pool2d(x, output_size=1)
        
        # Flatten pooled features
        B, C, _, _ = pooled.shape
        
        # Project to hidden dimension (>=640)
        memory = self.proj(pooled.view(B, C))
        
        return memory

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=6, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Initialize transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, 
                                                dim_feedforward=2048, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final projection layer to vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Learnable memory token (vocabulary-aware embedding)
        self.mem_emb = nn.Embedding(vocab_size, d_model)
        
    def forward(self, tgt, memory):
        # tgt: [B, T_prev]
        # memory: [B, S, d_model]
        
        # Embed target sequence
        embedded = self.embedding(tgt)
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
        
        # Create causal mask
        tgt_mask = self.create_causal_mask(tgt.shape[1]).to(tgt.device)
        
        # Apply transformer decoder with cross-attention
        out = self.transformer_decoder(
            embedded, 
            memory, 
            tgt_mask=tgt_mask
        )
        
        # Apply final projection to vocabulary
        logits = self.fc_out(out)
        
        return logits

class Net(nn.Module):
    def __init__(self, in_channels, out_channels, image_size=224, num_layers=6, num_heads=8, vocab_size=30522):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, out_channels, image_size, num_layers, num_heads)
        self.decoder = TransformerDecoder(vocab_size, out_channels, num_layers, num_heads)
        
    def train_setup(self, device, lr, momentum, **kwargs):
        self.device = device
        self.encoder.to(device)
        self.decoder.to(device)
        self.lr = lr
        self.momentum = momentum
        
    def learn(self, images, targets, **kwargs):
        # This method is intended to handle the training step
        # It should return a dictionary with loss and other metrics
        pass
        
    def forward(self, images, targets, **kwargs):
        # Teacher forcing: directly use the target sequence in the decoder
        memory = self.encoder(images)
        output = self.decoder(targets, memory)
        
        # Shape assertions
        assert images.dim() == 4, "Input images must be 4D tensors"
        assert images.shape[1] == in_channels, f"Expected {in_channels} channels, got {images.shape[1]}"
        assert images.shape[2] == images.shape[3] == image_size, f"Expected square image of size {image_size}, got ({images.shape[2]}, {images.shape[3]})"
        
        assert targets.dim() == 2, "Input targets must be 2D tensors"
        assert targets.shape[1] == self.decoder.vocab_size, f"Expected targets with vocabulary size {self.decoder.vocab_size}, got {targets.shape[1]}"
        
        return output