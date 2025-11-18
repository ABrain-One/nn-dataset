import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(start_dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0:6] = torch.arange(0, 6) / 6
        for i in range(1, d_model // 2 + 1):
            if i <= 6:
                div_term = torch.exp(torch.arange(-i, -i - d_model//2 + 1, -1) * (2 * math.pi * 0.5 ** (2 * (i - 1) // 2) / d_model))
            else:
                div_term = torch.exp(torch.arange(-i, -i - d_model//2 + 1, -1) * (2 * math.pi * 0.5 ** ((i - 1) % 2) / d_model))
            pe[0, :, 2*i-1] = torch.sin(position * div_term)
            pe[0, :, 2*i] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, x):
        # x shape: [B, T, d_model]
        seq_len = x.size(1)
        x = x * self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x)

class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

class TransformerDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, batch_first=True):
        super().__init__(decoder_layer, num_layers, batch_first)

class CNN_ENCODER(nn.Module):
    def __init__(self, input_channel=3, output_dim=640):
        super(CNN_ENCODER, self).__init__()
        self.output_dim = output_dim
        
        self.body = nn.Sequential(
            # Stem
            nn.Conv2d(input_channel, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Stage 1
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Stage 2
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Stage 3
            nn.Conv2d(128, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Stage 4
            nn.Conv2d(256, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Final pooling
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        h = self.body(x)
        return h.unsqueeze(1)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.hidden_dim = 640
        self.vocab_size = int(out_shape[0])
        self.encoder = CNN_ENCODER(input_channel=in_shape[1], output_dim=self.hidden_dim)
        num_layers = prm.get('num_layers', 6)
        nhead = min(prm.get('nhead', 8), self.hidden_dim // 32)
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model=self.hidden_dim, nhead=nhead),
            num_layers=num_layers
        )
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos_encoder = PositionalEncoding(self.hidden_dim)
        self.pos_decoder = PositionalEncoding(self.hidden_dim)
        self.criterion = nn.CrossEntropyLoss(ignore_index=prm.get('ignore_index', 0))
        self.to(device)

    def train_setup(self, prm):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'])

    def learn(self, train_data):
        self.train_setup({'lr': 0.0001})
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            memory = self.encoder(images)
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            embedded = self.embedding(inputs)
            embedded = embedded * math.sqrt(self.hidden_dim)
            embedded = self.pos_encoder(embedded)
            memory = memory.transpose(0, 1)  # [T, B, H]
            tgt_mask = torch.triu(torch.ones(embedded.size(0), embedded.size(0), dtype=torch.bool), diagonal=1)
            tgt_mask = torch.where(tgt_mask, float('-inf'), float(0.0)).to(self.device)
            logits = self.decoder(embedded, memory, tgt_mask=tgt_mask)
            logits = logits.transpose(0, 1)  # [B, T, H]
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        if captions is not None:
            memory = self.encoder(images)
            inputs = captions[:, :-1]
            embedded = self.embedding(inputs)
            embedded = embedded * math.sqrt(self.hidden_dim)
            embedded = self.pos_encoder(embedded)
            memory = memory.transpose(0, 1)
            tgt_mask = torch.triu(torch.ones(embedded.size(0), embedded.size(0), dtype=torch.bool), diagonal=1)
            tgt_mask = torch.where(tgt_mask, float('-inf'), float(0.0)).to(embedded.device)
            logits = self.decoder(embedded, memory, tgt_mask=tgt_mask)
            logits = logits.transpose(0, 1)
            return logits, None
        else:
            raise NotImplementedError("Beam search generation is not implemented in this version.")

def supported_hyperparameters():
    return {'lr', 'momentum'}