import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0] = torch.sin(position / torch.arange(max_len).float().unsqueeze(1) * torch.pi)
        pe[0, :, 1::2] = torch.cos(position / torch.arange(max_len).float().unsqueeze(1) * torch.pi)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x[:,:,:seq_len] + self.pe[:,:seq_len,:]
        return self.dropout(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 768, num_layers: int = 6, nhead: int = 8, 
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout, max_len=5000)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
            activation="gelu"
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim))
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(tgt)
        embedded = self.pos_encoding(embedded)
        seq_len = tgt.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = torch.where(mask == 1, float('-inf'), float(0)).to(tgt.device)
        out = self.transformer_decoder(embedded, memory, tgt_mask=mask)
        logits = self.fc_out(out)
        return logits, hidden_state

class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.stage1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.channel_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1, x.size(1))
        return x

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.vocab_size = out_shape[0]
        self.hidden_dim = 768
        self.encoder = Encoder(in_shape[1], self.hidden_dim)
        self.rnn = Decoder(self.vocab_size, self.hidden_dim, **prm)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def train_setup(self, **prm):
        self.rnn = self.rnn.to(self.device)
        self.encoder = self.encoder.to(self.device)

    def learn(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, **prm) -> torch.Tensor:
        images = images.to(self.device)
        captions = captions.to(self.device)
        memory = self.encoder(images)
        if captions is not None:
            inputs = captions[:, :-1]
            logits, _ = self.rnn(inputs, None, memory)
            assert logits.shape == (images.size(0), inputs.shape[1], self.vocab_size)
            return logits
        else:
            return None

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        images = images.to(self.device)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        memory = self.encoder(images)
        if captions is not None:
            inputs = captions[:, :-1]
            logits, hidden_state = self.rnn(inputs, hidden_state, memory)
            return logits, hidden_state
        else:
            return None, None

def supported_hyperparameters() -> set:
    return {'lr', 'momentum'}