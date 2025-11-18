import math
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape
        
        # Build encoder
        self.encoder = self.build_encoder(in_shape[1])
        
        # Build decoder
        self.decoder = self.build_decoder(prm, self.vocab_size)
        
        # Define other necessary components
        self.embed = nn.Embedding(self.vocab_size, 768)
        self.pos_encoder = self.build_positional_encoder(768)
        self.transformer = self.build_transformer_decoder(768, 8, 6)
        self.fc_out = nn.Linear(768, self.vocab_size)
        
    def build_encoder(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 768, 3, 1, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
    
    def build_decoder(self, prm, vocab_size):
        return nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=768, nhead=8, dim_feedforward=2048),
            num_layers=6
        )
    
    def build_positional_encoder(self, d_model):
        return nn.Parameter(torch.zeros(1, 1, d_model))
    
    def build_transformer_decoder(self, d_model, nhead, num_layers):
        return nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2048),
            num_layers=num_layers
        )
    
    def supported_hyperparameters():
    return {'lr','momentum'}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encoding
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x is expected to be [batch, seq, features]
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :, :].squeeze(1)
        x = x + pe
        return self.dropout(x)

def supported_hyperparameters():
    return {'lr', 'momentum'}