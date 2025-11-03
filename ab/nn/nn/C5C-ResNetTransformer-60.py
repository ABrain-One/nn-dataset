import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape)
        self.hidden_dim = 768  # Using 768 (greater than 640)
        
        # Encoder: Vision Transformer style feature extractor
        in_channels = int(in_shape[1])
        self.encoder = nn.Sequential(
            # Stem layer
            nn.Conv2d(in_channels, 192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Bottle-neck blocks with attention
            nn.LayerNorm([192, 14, 14]),  # Assuming 224->33->14->7 etc.
            SelfAttentionBlock(192, 6, 3),  # Example attention mechanism
            SelfAttentionBlock(256, 8, 3),
            SelfAttentionBlock(512, 16, 3),
            SelfAttentionBlock(1024, 32, 3),
            
            # Final projection
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, self.hidden_dim),
        )
        
        # Decoder: Transformer decoder with cross-attention
        num_heads = self.hidden_dim // 64  # Should divide evenly for standard MHSA
        self.decoder = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=2*self.hidden_dim,
            batch_first=True,
            enable_nested_tensor=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder, 
            num_layers=6,
            norm=nn.LayerNorm(self.hidden_dim)
        )
        
        # Output projection
        self.generator = nn.Linear(self.hidden_dim, self.vocab_size)
        
        # Embedding layer for vocabulary
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        
        # Positional encoding for transformer
        self.position_encoding = PositionalEncoding(self.hidden_dim)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.optimizer = None
        
        # Initialize weights according to BABBLER style
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2.0/fan_out)/np.sqrt(3))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)
    
    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.empty(0, device=device), torch.empty(0, device=device)
    
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
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            memory = self.encoder(images)
            embedded_inputs = self.token_embedding(inputs)
            embedded_inputs = self.position_encoding(embedded_inputs)
            
            # Add random noise during training for regularization
            noise = torch.randn_like(embedded_inputs) * 0.05
            embedded_inputs = embedded_inputs + noise
            
            # Check dimensions for transformer decoder
            torch._assert(embedded_inputs.shape[1] == memory.shape[1], 
                          f"Expected embedding sequence length {embedded_inputs.shape[1]} to match memory sequence length {memory.shape[1]}")
            torch._assert(embedded_inputs.shape[2] == memory.shape[2], 
                          f"Expected embedding feature dim {embedded_inputs.shape[2]} to match memory feature dim {memory.shape[2]}")
            
            # Prepare memory and embed for transformer decoder
            memory_with_padding = torch.zeros((images.size(0), memory.shape[1]+1, memory.shape[2]), device=self.device)
            memory_with_padding[:, -memory.shape[1]:, :] = memory
            
            # Transformer decoder needs key_padding_mask argument
            # Here we assume we pad with zeros at the beginning of memory
            tgt_mask = torch.lower_triangular(torch.full((targets.shape[1], targets.shape[1]), -float('Inf'), device=self.device))
            
            # Generate initial masks with padding
            memory_mask = torch.where(memory_with_padding.abs().sum(dim=2) > 0, 
                                    torch.ones_like(memory_with_padding), 
                                    torch.zeros_like(memory_with_padding))
            
            outputs = self.transformer_decoder(embedded_inputs, memory_with_padding, tgt_mask=tgt_mask, memory_key_padding_mask=~memory_mask.bool())
            
            # Apply final linear layer
            logits = self.generator(outputs)
            
            assert logits.shape == (images.size(0), inputs.shape[1], self.vocab_size), \
                   f"Logits shape {(images.size(0), inputs.shape[1], self.vocab_size)} does not match actual shape {logits.shape}"
            
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
    
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        
        if captions is not None:
            # Teacher forcing implementation
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            embedded = self.token_embedding(inputs)
            embedded = self.position_encoding(embedded)
            
            # If needed, initialize hidden_state manually though transformer doesn't maintain traditional hidden states
            # Instead, we rely solely on memory inputs and outputs
            
            # Compute tgt_mask and memory_mask
            tgt_mask = torch.lower_triangular(torch.full((inputs.shape[1], inputs.shape[1]), -float('Inf'), device=self.device))
            memory_mask = torch.where(memory.abs().sum(dim=2) > 0, 
                                     torch.ones_like(memory), 
                                     torch.zeros_like(memory))
            
            # Call transformer decoder
            outputs = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask)
            logits = self.generator(outputs)
            
            return logits, memory[:,-1,:]  # Return both logits and the last memory feature as new context
            
        else:
            # Beam search initialization
            raise NotImplementedError()

# Standard positional encoding for transformers
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].repeat(x.size(0), 1, 1)
        return self.dropout(x + pe)
    
# Simplified self-attention block for bottlenecks
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads, hidden_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_channels)
        self.multi_head_attention = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.linear = nn.Linear(in_channels, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Input: [batch, ..., channels]
        # First project to standard MHSA format
        ln_x = self.layer_norm(x)
        attn_output, _ = self.multi_head_attention(ln_x, ln_x, ln_x)
        return self.relu(self.linear(attn_output))