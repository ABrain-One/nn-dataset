import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        pe = torch.zeros(max_len, 1, d_model)
        pe[0, 0, 0] = 1
        for pos in range(1, max_len):
            pe[pos, 0, 0] = 0.1 * torch.sin(pos / (max_len ** 0.5) * (2 * torch.pi))
            pe[pos, 0, 1] = 0.1 * torch.cos(pos / (max_len ** 0.5) * (2 * torch.pi))
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MyEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=768):
        super(MyEncoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU,
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU,
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU,
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU,
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU,
        )
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x.unsqueeze(1)

class MyDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768, num_layers=6, nhead=8):
        super(MyDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=2048, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, memory):
        embedded = self.embedding(tgt)
        embedded = self.pos_encoding(embedded)
        seq_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
        tgt_mask = torch.where(tgt_mask, float('-inf'), float(0.0))
        out = self.transformer_decoder(
            embedded, 
            memory,
            tgt_mask=tgt_mask,
        )
        return self.fc_out(out)

class Net(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.encoder = MyEncoder(hidden_dim=hyperparameters['hidden_dim'])
        self.decoder = MyDecoder(
            vocab_size=hyperparameters['vocab_size'],
            hidden_dim=hyperparameters['hidden_dim'],
            num_layers=hyperparameters['num_layers'],
            nhead=hyperparameters['nhead']
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def train_setup(self, train_data):
        pass

    def learn(self, images, captions):
        images = images.to(self.device)
        captions = captions.to(self.device)
        memory = self.encoder(images)
        tgt_input = captions[:, :-1]
        logits, _ = self.decoder(tgt_input, memory)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1))
        return loss

    def forward(self, images, captions=None, hidden_state=None):
        if captions is not None:
            if isinstance(captions, torch.Tensor):
                if captions.ndim == 3:
                    tgt_input = captions[:, :-1]
                    targets = captions[:, 1:]
                    memory = self.encoder(images)
                    embedded = self.decoder.embedding(tgt_input)
                    embedded = self.decoder.pos_encoding(embedded)
                    seq_len = tgt_input.size(1)
                    tgt_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
                    tgt_mask = torch.where(tgt_mask, float('-inf'), float(0.0))
                    logits = self.decoder.transformer_decoder(
                        embedded,
                        memory,
                        tgt_mask=tgt_mask
                    )
                    logits = self.decoder.fc_out(logits)
                    return logits, None
                elif captions.ndim == 2:
                    tgt_input = captions[:, :-1]
                    targets = captions[:, 1:]
                    memory = self.encoder(images)
                    embedded = self.decoder.embedding(tgt_input)
                    embedded = self.decoder.pos_encoding(embedded)
                    seq_len = tgt_input.size(1)
                    tgt_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
                    tgt_mask = torch.where(tgt_mask, float('-inf'), float(0.0))
                    logits = self.decoder.transformer_decoder(
                        embedded,
                        memory,
                        tgt_mask=tgt_mask
                    )
                    logits = self.decoder.fc_out(logits)
                    return logits, None
            else:
                raise ValueError(f"Captions must be a torch.Tensor, got {type(captions)}")
        else:
            raise ValueError("Captions must be provided for training")

    @property
    def device(self):
        return next(self.parameters()).device

def supported_hyperparameters():
    return {'lr', 'momentum', 'hidden_dim', 'num_layers', 'nhead', 'vocab_size'}