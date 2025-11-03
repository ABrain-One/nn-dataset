import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Label

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return x

class CNN_Encoder(nn.Module):
    def __init__(self, in_channels: int, out_features: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, out_features)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.cnn(images).unsqueeze(1)

class Transformer_Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 768, num_layers: int = 6, nhead: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tgt)
        embedded = self.pos_encoding(embedded)
        mask = self.generate_square_subsequent_mask(tgt.size(1)).to(memory.device)
        return self.transformer_decoder(embedded, memory, mask)

    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        mask = torch.triu(torch.full((sz, sz), float("-Inf")), diagonal=1)
        return mask.type_as(next(self.parameters()))

def supported_hyperparameters():
    return {'lr','momentum'}


    def init_zero_hidden(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.empty(0, dtype=torch.float, device=device),) * 2

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=max(float(prm.get('lr', 1e-3)), 3e-4))
        self.criterion = nn.CrossEntropyLoss(ignore_index=prm.get('pad_idx', 0))

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions[:, 0, :].long().to(self.device)
            memory = self.encoder(images)
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            
            outputs = self.decoder(inputs, memory)
            loss = self.criterion(outputs.reshape(-1, self.vocab_size), targets.reshape(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None) -> tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device)
        memory = self.encoder(images)
        
        if captions is not None:
            if captions.ndim == 3:
                captions = captions[:, 0, :]
                
            inputs = captions[:, :-1]
            outputs = self.decoder(inputs, memory)
            assert outputs.shape == (images.size(0), captions.shape[1]-1, self.vocab_size)
            assert memory.shape == (images.size(0), 1, self.hidden_dim)
            return outputs, memory
        
        return memory