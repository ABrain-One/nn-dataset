import torch.nn as nn
import torch.optim as optim
import torch
import math
from typing import Tuple, Dict, Any

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(d_model).float() * -math.log(10000.0) / d_model)
        pe = torch.zeros((max_len, 1, d_model))
        pe[:, 0, 0::2] = position.float(). view(-1, d_model//2) * div_term[0::2]
        pe[:, 0, 1::2] = position.float(). view(-1, d_model//2) * div_term[1::2]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be [B, T, D]
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class Net(nn.Module):
    def __init__(self, in_shape, hidden_dim=768, **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.hidden_dim = hidden_dim
        
        # Encoder: CNN that outputs [B, 1, 768]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 768, kernel_size=1)
        )
        
        # Decoder: GRU that takes [B, T] and [B, 1, 768] and outputs [B, T, hidden_dim]
        self.embedding = nn.Embedding(kwargs['vocab_size'], hidden_dim)
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, kwargs['vocab_size'])
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
    def train_setup(self, params, prm):
        # Use AdamW with learning rate and momentum from prm
        optimizer = optim.AdamW(params, lr=prm['lr'], weight_decay=prm['momentum'])
        return optimizer, None
        
    def learn(self, train_data):
        # train_data is a tuple (images, captions)
        images, captions = train_data
        
        # Forward pass through encoder
        memory = self.encoder(images)
        
        # Prepare input for decoder
        inputs = captions[:, :-1]
        targets = captions[:, 1:]
        
        # Embed inputs
        embedded = self.embedding(inputs)
        
        # Expand memory to match input sequence length
        memory = memory.expand(inputs.size(0), inputs.size(1), -1)
        
        # Concatenate embedded input and memory
        combined = torch.cat([embedded, memory], dim=-1)
        
        # Forward pass through decoder
        output, _ = self.gru(combined)
        
        # Final output layer
        logits = self.fc_out(output)
        
        # Calculate loss
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
        
        # Backward pass and optimizer step
        optimizer = self.train_setup(self.parameters(), prm)[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
        
    def forward(self, images, captions=None, hidden_state=None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.size(0)
        
        # Encoder forward pass
        memory = self.encoder(images)
        
        if captions is not None:
            # Decoder forward pass
            inputs = captions[:, :-1]
            embedded = self.embedding(inputs)
            memory = memory.expand(inputs.size(0), inputs.size(1), -1)
            combined = torch.cat([embedded, memory], dim=-1)
            
            output, hidden_state = self.gru(combined, hidden_state)
            logits = self.fc_out(output)
            
            # Reshape logits for loss calculation
            logits = logits.reshape(-1, logits.size(-1))
            targets = captions[:, 1:].reshape(-1)
            
            loss = self.criterion(logits, targets)
            return loss, logits, hidden_state
        else:
            # If captions are None, we are in evaluation mode and need to return memory
            return memory, None, None

def supported_hyperparameters() -> Dict[str, Any]:
    return {'lr': 0.001, 'momentum': 0.001}