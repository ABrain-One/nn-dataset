import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.view(max_len, d_model).unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Net(nn.Module):
    def __init__(self, in_channels=3, out_dim=768, vocab_size=10000, d_model=768, nhead=8, num_layers=6):
        super(Net, self).__init__()
        # Encoder
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.projection = nn.Linear(1024, out_dim)
        
        # Decoder
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers, batch_first=True)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.hidden_dim = out_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.d_model = d_model
        
    def train_setup(self, images, captions):
        pass
        
    def learn(self, images, captions):
        pass
        
    def forward(self, images, captions=None):
        if captions is not None:
            # Encoder
            x = self.initial_conv(images)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.global_avgpool(x)
            memory = self.projection(x)  # [B, 1, hidden_dim]
            
            # Decoder
            tgt = captions[:, :-1]  # [B, T-1]
            embedded = self.embedding(tgt)
            embedded = self.pos_encoding(embedded)
            out = self.transformer_decoder(embedded, memory)
            logits = self.fc_out(out)
            
            return logits, None
        else:
            # Encoder
            x = self.initial_conv(images)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.global_avgpool(x)
            memory = self.projection(x)  # [B, 1, hidden_dim]
            
            # Decoder
            embedded = torch.zeros((images.size(0), captions.shape[1], self.hidden_dim))
            embedded = self.pos_encoding(embedded)
            out = self.transformer_decoder(embedded, memory)
            logits = self.fc_out(out)
            
            return logits, None

def supported_hyperparameters():
    return {'lr','momentum'}