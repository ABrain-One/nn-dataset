import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1, 0, 2)  # [B, S, H]
        seq_len = x.size(1)
        pe = self.pe[:seq_len]
        x = x + pe
        x = x.permute(1, 0, 2)  # back to [S, B, H]
        return self.dropout(x)

class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float, attention_dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attention_dropout = attention_dropout
        self.dropout = nn.Dropout(dropout)
        
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * mlp_ratio)
        self.dropout_mlp = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim * mlp_ratio, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm1(x)
        x = F.softmax(self.attention(x, x, x)[0], dim=1)
        x = self.dropout(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout_mlp(x)
        x = self.linear2(x)
        x = self.layernorm2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_chans: int, hidden_dim: int, num_layers: int, num_heads: int, mlp_ratio: float, dropout: float, attention_dropout: float):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim)
        )
        self.blocks = nn.ModuleList([
            EncoderBlock(hidden_dim, num_heads, mlp_ratio, dropout, attention_dropout) 
            for _ in range(num_layers)
        ])
        self.to_sequence = nn.Flatten(start_dim=2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.stem(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), -1, self.hidden_dim * (x.size(3) // self.stem[0].kernel_size[1]))
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x.permute(0, 2, 1)  # [B, H, S]
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)  # [B, S, H]
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, vocab_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        tgt = tgt.to(self.device)
        memory = memory.to(self.device)
        x = self.embedding(tgt)
        x = self.pos_encoding(x)
        x = self.layernorm(x)
        for block in self.blocks:
            x = block(x, memory)
        x = self.fc_out(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int, int], out_shape: Tuple[int, ...], 
                 prm: Dict, device: torch.device):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device
        
        # Extract parameters
        in_chans = in_shape[1]
        height = in_shape[2]
        width = in_shape[3]
        hidden_dim = self.prm.get('hidden_dim', 768)
        num_layers = self.prm.get('num_layers', 6)
        num_heads = self.prm.get('num_heads', 8)
        mlp_ratio = self.prm.get('mlp_ratio', 4)
        dropout = self.prm.get('dropout', 0.1)
        attention_dropout = self.prm.get('attention_dropout', 0.1)
        vocab_size = out_shape[0]
        
        self.encoder = Encoder(in_chans, hidden_dim, num_layers, num_heads, mlp_ratio, dropout, attention_dropout)
        self.decoder = Decoder(hidden_dim, num_heads, num_layers, vocab_size)
        self.to_logits = nn.Linear(hidden_dim, vocab_size)

    def train_setup(self, lr: float, momentum: float):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def learn(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        images = images.to(self.device)
        if captions is not None:
            captions = captions.to(self.device)
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            memory = self.encoder(images)
            logits = self.decoder(inputs, memory)
            logits = self.to_logits(logits)
            loss = self.criterion(logits.reshape(-1, self.out_shape[0]), targets.reshape(-1))
            return logits, loss
        else:
            memory = self.encoder(images)
            return memory

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        images = images.to(self.device)
        if captions is not None:
            captions = captions.to(self.device)
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            memory = self.encoder(images)
            logits = self.decoder(inputs, memory)
            logits = self.to_logits(logits)
            return logits, None
        else:
            memory = self.encoder(images)
            return memory

def supported_hyperparameters():
    return {'lr', 'momentum'}