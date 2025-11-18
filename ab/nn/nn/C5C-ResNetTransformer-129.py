

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def supported_hyperparameters():
    return {'lr','momentum'}


class AttentionPooling(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        
        # Pre-calculate positional embedding if needed later
        # Implementation assumes query and key are from same sequence
        # but technically this is more complex than shown here
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Normalize values
        q = q * self.scale
        
        # Calculate attention scores
        attn_scores = (q @ k.transpose(-2,-1)) / self.scale
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        x = (attn_weights @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.mean(dim=1)

class CaptionEncoder(nn.Module):
    def __init__(self, in_chans=3, dim=768, depth=6, heads=12, qkv_bias=True, 
                 dropout=0.1, patch_size=16, device='cuda'):
        super().__init__()
        # Patch creation layer
        self.patch_embedding = nn.Conv2d(in_chans, dim, kernel_size=patch_size)
        
        # Number of tokens from grid
        grid_size = (in_shape[2] // patch_size) ** 2 if hasattr(in_shape,'image') else 1
        
        # Learnable class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim*4,
                batch_first=True, dropout=dropout
            ),
            num_layers=depth
        )
        
        self.grid_size = grid_size
        self.to_device(device)
        
    def forward(self, x):
        # Get patches
        x = self.patch_embedding(x)  # Shape [B, C, H, W]
        
        # Calculate grid tokens
        grid_size = x.shape[2] * x.shape[3]  # Assuming all same size
        x = x.permute(0, 2, 3, 1).contiguous().view(x.shape[0], grid_size, self.dim) 
        
        # Append class token
        class_token = self.class_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([class_token, x], dim=1)
        
        # Process through transformer
        x = self.transformer(x)
        
        # Return either just class token or the whole sequence
        return x[: ,0]  # Just class token, which captures contextual info

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, hidden_dim=768, num_layers=1, dropout=0.):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=False, dropout=int(dropout>0)
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        
    def forward(self, inputs, hidden_state, memory):
        # Embed inputs
        embedded = self.embedding(inputs)
        embedded = embedded + memory
        
        # Initial hidden state
        if hidden_state is None:
            batch_size = inputs.shape[0]
            hidden_state = self.init_hidden(batch_size)
        
        # Run through RNN
        output, hidden_state = self.rnn(embedded, hidden_state)
        
        # Final projection
        logits = self.fc(output)
        return logits, hidden_state
    
    def init_hidden(self, batch_size):
        return (
            torch.randn(self.num_layers, batch_size, self.hidden_dim, device='cuda'),
            torch.randn(self.num_layers, batch_size, self.hidden_dim, device='cuda')
        )

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm={}, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        self.in_shape = in_shape  # (C, H, W) or (B, C, H, W) but processed as (B, ...)
        self.out_shape = out_shape  # (vocab_size,) assumed
        self.vocab_size = out_shape[0]
        self.dim = int(prm.get('dimension', 768))  # Should be ≥640
        
        # Encoder setup
        self.encoder = CaptionEncoder(
            in_chans=in_shape[1][-3] if isinstance(in_shape, list) else in_shape[1],
            dim=self.dim,
            depth=min(int(prm.get('blocks', 6)), 12),  # Limit to at most 12 layers
            heads=min(int(prm.get('heads', 12)), self.dim//64) if self.dim%64==0 else min(int(prm.get('heads', 12)), self.dim//32),
            qkv_bias=bool(prm.get('qkv_bias', True)),
            dropout=float(prm.get('dropout', 0.1)) if isinstance(prm, dict) else 0.1,
            patch_size=int(prm.get('patch_size', 16)) if hasattr(prm, 'get') else 16
        )
        
        # Decoder setup
        self.decoder = LSTMDecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.dim,
            hidden_dim=self.dim,
            num_layers=min(int(prm.get('rnn_layers', 1)), 2),
            dropout=float(prm.get('dropout', 0.1)) if isinstance(prm, dict) else 0.1
        )
        
        # Optimizer and loss
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        
    def init_zero_hidden(self, batch, device):
        return (
            torch.zeros(self.decoder.num_layers, batch, self.decoder.hidden_dim, 
                       device=device),
            torch.zeros(self.decoder.num_layers, batch, self.decoder.hidden_dim, 
                       device=device)
        )
    
    def train_setup(self, prm={}):
        self.to(self.device)
        
        # Learning rate and momentum handling
        lr_val = float(prm.get('lr', 1e-3))
        momentum_val = float(prm.get('momentum', 0.9))
        
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr_val, betas=(momentum_val, 0.999)
        )
        
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            cap_input = captions[:, :-1] if captions.ndim == 3 else captions[:, :-1, :]
            cap_target = captions[:, 1:] if captions.ndim == 3 else captions[:, 1:, :]
            
            # Forward pass
            memory = self.encoder(images)
            logits, _ = self.decoder(cap_input, None, memory)
            
            # Loss calculation
            loss = self.criterion(
                logits.view(-1, self.vocab_size),
                cap_target.contiguous().view(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                float(prm.get('grad_clip', 3.0)) if isinstance(prm,dict) else 3.0
            )
            self.optimizer.step()
            
    def forward(self, images, captions=None, hidden_state=None):
        images = images.to(self.device)
       