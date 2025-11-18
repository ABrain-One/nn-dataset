import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(d_model) * -torch.log(torch.tensor(10000.0)) * 2)
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position / div_term)
        pe[:, 0, 1::2] = torch.cos(position / div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_dim=768, num_layers=6, nhead=8, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, hidden_dim, kernel_size=3, stride=1, padding=1)
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True, dropout=dropout),
            num_layers=num_layers
        )
        self.embedding = nn.Embedding(out_shape, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        self.fc_out = nn.Linear(hidden_dim, out_shape)
        self.device = None

    def train_setup(self, optimizer, lr, momentum):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            param_group['momentum'] = momentum

    def learn(self, images, captions=None, hidden_state=None):
        if captions is not None:
            memory = self.encoder(images)
            if isinstance(captions, torch.Tensor):
                if captions.dim() == 3:
                    caps = captions[:, 0, :].long().to(self.device)
                else:
                    caps = captions.long().to(self.device)
            else:
                raise ValueError("Captions must be a torch.Tensor")
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            embedded = self.embedding(inputs)
            embedded = self.pos_encoder(embedded)
            tgt_mask = self.generate_square_subsequent_mask(embedded.size(1), embedded.device)
            output = self.decoder(tgt=embedded, memory=memory, tgt_mask=tgt_mask)
            logits = self.fc_out(output)
            loss = F.cross_entropy(logits.view(-1, out_shape), targets.view(-1))
            return loss
        return None

    def forward(self, images, captions=None, hidden_state=None):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        if captions is not None:
            if isinstance(captions, torch.Tensor):
                if captions.dim() == 3:
                    caps = captions[:, 0, :].long().to(self.device)
                else:
                    caps = captions.long().to(self.device)
            else:
                raise ValueError("Captions must be a torch.Tensor")
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            embedded = self.embedding(inputs)
            embedded = self.pos_encoder(embedded)
            tgt_mask = self.generate_square_subsequent_mask(embedded.size(1), embedded.device)
            output = self.decoder(tgt=embedded, memory=memory, tgt_mask=tgt_mask)
            logits = self.fc_out(output)
            if hidden_state is None:
                hidden_state = output[:,-1,:]
            return logits, hidden_state
        else:
            return self.decoder(None, None, None)

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf')).to(device)
        return mask

def supported_hyperparameters():
    return {'lr','momentum'}