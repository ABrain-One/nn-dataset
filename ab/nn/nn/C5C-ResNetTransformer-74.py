import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.view(-1, d_model).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:x.size(0), :x.size(1), :])

class Net(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], out_shape: Tuple[int, ...], 
                 device: torch.device, vocab_size: int = None):
        super(Net, self).__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.vocab_size = out_shape[0] if out_shape else vocab_size
        
        # Encoder: CNN that outputs [B, S, H] with H>=640
        self.encoder_cnn = self.build_encoder(in_shape[1], in_shape[2])
        self.flatten = nn.Flatten(start_dim=2)
        self.proj = nn.Linear(512, 768)
        
        # Decoder: Transformer decoder
        self.embedding_dim = 768
        self.decoder_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.positional_encoding = PositionalEncoding(self.embedding_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.embedding_dim, nhead=8),
            num_layers=6, batch_first=True
        )
        self.fc_out = nn.Linear(self.embedding_dim, self.vocab_size)
        
    def build_encoder(self, channels: int, size: int) -> nn.Sequential:
        layers = []
        # Calculate intermediate sizes
        sizes = [size // (2**4)]
        
        layers.append(nn.Conv2d(channels, 64, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU)
        layers.append(nn.MaxPool2d(2))
        
        layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU)
        layers.append(nn.MaxPool2d(2))
        
        layers.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU)
        layers.append(nn.MaxPool2d(2))
        
        layers.append(nn.Conv2d(256, 512, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU)
        
        self.encoder_cnn = nn.Sequential(*layers)
        return self.encoder_cnn
    
    def train_setup(self, **kwargs):
        pass
    
    def learn(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, 
              hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if captions is not None:
            memory = self.encode(images)
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            tgt_len = inputs.size(1)
            memory_len = memory.size(1)
            
            # Project memory to embedding dimension
            memory = memory.permute(1, 0, 2)  # [S, B, H]
            
            # Embed inputs and apply positional encoding
            tgt_emb = self.decoder_embedding(inputs)
            tgt_emb = self.positional_encoding(tgt_emb)
            
            # Forward pass through transformer decoder
            output = self.transformer_decoder(tgt_emb, memory)
            logits = self.fc_out(output)
            
            # Calculate loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            
            # Return loss and memory (for teacher forcing)
            return loss, logits, memory
            
        else:
            # For inference, return memory and hidden_state
            memory = self.encode(images)
            return memory, None
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self.encoder_cnn(images)
        x = self.flatten(x)
        x = self.proj(x)
        x = x.permute(0, 2, 1)  # [B, H, S]
        return x
    
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, 
                hidden_state: Optional[torch.Tensor] = None):
        if captions is not None:
            # Teacher forcing
            memory = self.encode(images)
            tgt_emb = self.decoder_embedding(captions[:, :-1])
            tgt_emb = self.positional_encoding(tgt_emb)
            memory = memory.permute(1, 0, 2)
            output = self.transformer_decoder(tgt_emb, memory)
            logits = self.fc_out(output)
            return logits, hidden_state
            
        else:
            # Beam search or other generation method
            memory = self.encode(images)
            memory = memory.permute(1, 0, 2)
            return self.transformer_decoder(None, memory)

def supported_hyperparameters():
    return {'lr','momentum'}