import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(d_model, dtype=torch.float) * - (torch.log(torch.tensor(1000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0:d_model//2] = torch.sin(position[:, 0, 0:d_model//2] / (1000.0 ** (2.0 * torch.arange(0, d_model//2, dtype=torch.float) / d_model)))
        pe[:, 0, d_model//2:d_model] = torch.cos(position[:, 0, 0:d_model//2] / (1000.0 ** (2.0 * torch.arange(0, d_model//2, dtype=torch.float) / d_model)))
        pe = pe[:, 0, :].unsqueeze(0)  # shape [1, max_len, d_model]
        pe = pe.repeat(batch_size, 1, 1)  # shape [batch_size, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:x.size(0), :])

class MyViTEncoder(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], out_shape: Tuple[int, int], prm, device):
        super().__init__()
        self.device = device
        self.num_patches = (in_shape[1] // 16) * (in_shape[2] // 16)
        self.embed = nn.Linear(3, 768)  # Simple projection to match the example
        self.pos_encoding = PositionalEncoding(768)
        num_layers = 6
        num_heads = 8
        hidden_dim = 768
        dropout_rate = prm['dropout'] if 'dropout' in prm else 0.1
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=3072, dropout=dropout_rate, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Project to out_shape[0] (vocab_size) if needed
        self.projection = nn.Linear(hidden_dim, out_shape[0])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.size(0)
        # images: [B, C, H, W]
        # Flatten the spatial dimensions: [B, C, H*W]
        features = images.view(batch_size, 3, -1)
        # Project to hidden_dim
        features = self.embed(features)
        # Add positional encoding
        features = self.pos_encoding(features)
        # Transformer encoding
        memory = self.encoder(features)
        # Project to vocab_size if needed
        memory = self.projection(memory)
        return memory

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape[0]
        self.embed = nn.Embedding(self.vocab_size, 768)
        self.pos_encoding = PositionalEncoding(768)
        num_layers = 6
        num_heads = 8
        hidden_dim = 768
        dropout_rate = prm['dropout'] if 'dropout' in prm else .1
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=3072, dropout=dropout_rate, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=3072, dropout=dropout_rate, batch_first=True),
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(hidden_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def init_zero_hidden(self, batch_size):
        return None

    def train_setup(self, batch_size):
        pass

    def learn(self, train_data):
        images, captions = train_data
        memory = self.encoder(images)
        inputs = captions[:, :-1]
        targets = captions[:, 1:]
        embedded = self.embed(inputs)
        embedded = self.pos_encoding(embedded)
        outputs = self.decoder(embedded, memory)
        logits = self.fc(outputs)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), reduction='mean')
        return loss

    def forward(self, images, captions=None, hidden_state=None):
        batch_size = images.size(0)
        if captions is not None:
            inputs = captions[:, :-1]
            embedded = self.embed(inputs)
            embedded = self.pos_encoding(embedded)
            memory = self.encoder(images)
            outputs = self.decoder(embedded, memory)
            logits = self.fc(outputs)
            return (logits, None)
        else:
            # Inference code would go here
            pass

def supported_hyperparameters():
    return {'lr', 'momentum'}