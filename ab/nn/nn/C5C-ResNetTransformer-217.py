import math
import torch
import torch.nn as nn
import torch.nn.functional as F
def supported_hyperparameters():
    return {'lr','momentum'}


class _Encoder(nn.Module):
    def __init__(self, channels, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(hidden_dim)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(d_model) * -math.log(10000.0) / d_model)
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, :] = position.repeat(1, d_model).unsqueeze(-1)[:, :, 0] * div_term
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        
        # Configuration
        self.vocab_size = out_shape
        self.hidden_dim = 768
        
        # Encoder configuration
        encoder_channels = 64
        self.encoder = _Encoder(encoder_channels, self.hidden_dim)
        
        # Decoder configuration
        decoder_num_layers = 6
        decoder_nhead = 8
        
        # Decoder
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=decoder_nhead,
            d_hid=self.hidden_dim,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_num_layers)
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim)
    
    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.zeros(1, batch, self.hidden_dim, device=device)
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999))

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images, captions = images.to(self.device), captions.to(self.device)
            logits = None
            if hasattr(self, 'forward'):
                out = self.forward(images, captions)
                logits = out[0] if isinstance(out, tuple) else out
            if logits is None:
                continue
            tgt = (captions[:,0,:] if captions.ndim==3 else captions)[:,1:]
            loss = self.criteria[0](logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()