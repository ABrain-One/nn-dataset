import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

class Net(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], out_shape: int, prm: Dict[str, Any], device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape
        self.hidden_dim = 768  # Ensure hidden dimension >= 640

        # Encoder using modified ResNet50 backbone
        self.encoder = self.build_encoder(in_shape[1], in_shape[1], in_shape[2], self.hidden_dim)
        
        # Decoder using Transformer
        self.decoder = self.build_decoder(self.hidden_dim, self.vocab_size)

    def build_encoder(self, input_channels, height, width, hidden_dim):
        # Simplified CNN encoder to produce memory features
        encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # Additional layers to reach desired hidden_dim
            nn.Flatten(),
            nn.Linear(64 * (width//4) * (height//4), hidden_dim)
        )
        return encoder

    def build_decoder(self, hidden_dim, vocab_size):
        # Transformer decoder
        embedding = nn.Embedding(vocab_size, hidden_dim)
        positional_encoding = PositionalEncoding(hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=3, batch_first=True)
        projection = nn.Linear(hidden_dim, vocab_size)
        return nn.Sequential(
            embedding,
            positional_encoding,
            decoder,
            projection
        )

    def train_setup(self, optimizer):
        # Set up for training
        pass

    def learn(self, images, captions):
        # Training step
        memory = self.encoder(images)
        if captions.ndim == 3:
            caps = captions[:,0,:].long().to(self.device)
        else:
            caps = captions.long().to(self.device)
        inputs = caps[:, :-1]
        targets = caps[:, 1:]
        out = self.decoder(inputs, memory)
        logits = out @ torch.arange(self.vocab_size).to(self.device)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return loss

    def forward(self, images, captions=None):
        # Forward pass
        if captions is not None:
            if captions.ndim == 3:
                caps = captions[:,0,:].long().to(self.device)
            else:
                caps = captions.long().to(self.device)
            inputs = caps[:, :-1]
            memory = self.encoder(images)
            out = self.decoder(inputs, memory)
            logits = out @ torch.arange(self.vocab_size).to(self.device)
            return logits, None
        else:
            raise NotImplementedError()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.pow(10000.0, torch.arange(0, d_model, 2.0) / d_model)
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position / div_term)
        pe[:, 0, 1::2] = torch.cos(position / div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is expected to be [batch_size, seq_length, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:seq_len]
        return self.dropout(x)

def supported_hyperparameters():
    return {'lr','momentum'}