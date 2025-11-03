import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:max_len, 0, 0::2] = torch.sin(position[:max_len, 0] * div_term)
        pe[:max_len, 0, 1::2] = torch.cos(position[:max_len, 0] * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        self.dropout = nn.Dropout(p=0.1)
        x = self.dropout(x)
        return self.pe[:seq_len, :, :].detach() + x
    
class CNNEncoder(nn.Module):
    def __init__(self, input_channels: int, output_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, output_dim, 1, 1, 0)
        ).requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: (B, C, H, W) → (B, output_dim, 1, 1)
        return self.net(x).flatten(start_dim=1).unsqueeze(1)

class CustomDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.layernorm3 = nn.LayerNorm(hidden_size)
        
    def forward(self, tgt: torch.Tensor, memory: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tgt = self.layernorm1(tgt + F.pad(self.self_attn(tgt, tgt, tgt)[0], (0, 0, 0, 0, 0, self.tgt_len - tgt.size(1)), "constant", 0))
        
        if memory is not None:
            memory_padded = F.pad(memory, (0, 0, 0, self.tgt_len - memory.size(1)), "constant", 0)
            tgt = self.layernorm2(tgt + self.cross_attn(tgt, memory_padded, memory_padded, key_padding_mask=key_padding_mask))
        
        tgt = self.layernorm3(tgt + self.ffn(tgt))
        return tgt

class ImageCaptioningModel(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: Dict, device: torch.device):
        super().__init__()
        self.device = device
        
        # Get hyperparameters from prm dictionary
        input_channels = int(in_shape[1])
        output_dim = max(int(out_shape), 768)  # ≥640, default 768 if smaller
        self.vocab_size = int(out_shape)
        self.lr = float(prm.get('lr', 1e-3))
        self.momentum = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        
        # Encoder setup
        self.encoder = CNNEncoder(input_channels, output_dim)
        
        # Decoder setup
        num_layers = prm.get('num_layers', 6)
        num_heads = min(prm.get('num_heads', 8), output_dim // 4)  # Divide hidden_size, e.g., 768→8/12
        hidden_size = output_dim
        
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_layers.append(CustomDecoderLayer(hidden_size, num_heads, 0.1))
            
        # Text embedding layer
        self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def init_zero_hidden(self, batch: int, device: torch.device):
        # Return empty tensors for encoder and decoder hidden states
        return (torch.empty(0, dtype=torch.float, device=device), 
                torch.empty(0, dtype=torch.float, device=device))

    def train_setup(self, prm):
        self.to(self.device)
        
        # Adjust hyperparameters
        self.lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        self.beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), 
                                          lr=self.lr, 
                                          betas=(self.beta1, 0.999))
        
        # Loss criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        
        # Move to device
        self.to(self.device)
        self.criterion = self.criterion.to(self.device)

    def learn(self, train_data):
        self.train()
        
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:, 0, :].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            # Compute memory features
            memory = self.encoder(images)
            # Teacher forcing
            if inputs.dim() == 2:  # [B, S]
                inputs = inputs.unsqueeze(1)  # [B, 1, S]
            embedded = self.embedding(inputs) + self.pos_encoder(embedded)
            
            # Process through decoder layers
            decoder_output = embedded
            for layer in self.decoder_layers:
                decoder_output = layer(decoder_output)
                
            # Final output layer
            logits = decoder_output.transpose(1, 2)  # [B, S, H] → [B, S, V]
            logits = F.linear(logits, None, None)
            
            # Calculate loss
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        
        if captions is not None:
            caps = captions[:, 0, :].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1] if captions.ndim == 3 else caps
            
            # Get embedding
            embedded = self.embedding(inputs)
            embedded = embedded + self.pos_encoder(embedded)
            
            # Feed to decoder
            decoder_output = embedded
            for layer in self.decoder_layers:
                decoder_output = layer(decoder_output, memory.squeeze(1))
                
            # Final linear layer
            logits = decoder_output.transpose(1, 2)
            return logits, None
            
        else:
            # For generation (use identity for decoder if captions aren't provided)
            return self.embed(images), None

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        # Base embedding without captions
        memory = self.encoder(images)
        embedded = self.embedding(torch.tensor([[self.word2idx.get('<SOS>', 0)]])).squeeze(1)
        embedded = embedded + self.pos_encoder(embedded)
        return embedded

def supported_hyperparameters():
    return {'lr','momentum'}
