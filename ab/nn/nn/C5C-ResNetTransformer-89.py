import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = position.float()[:, None] * div_term
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 768, num_layers: int = 6, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=2048)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, inputs: torch.Tensor, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(inputs)
        embedded = self.pos_encoder(embedded)
        out = self.transformer_decoder(embedded, memory=memory)
        logits = self.projection(out)
        return logits, out[:, -1, :]

class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 768):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        self.layer1 = self._make_layer(in_channels, 64)
        self.layer2 = self._make_layer(64, 128)
        self.layer3 = self._make_layer(128, 256)
        self.layer4 = self._make_layer(256, 512)
        self.layer5 = self._make_layer(512, hidden_dim)
        
    def _make_layer(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.hidden_dim)
        return x

class Net(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], out_shape: int, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.hidden_dim = 768
        self.encoder = CNNEncoder(in_shape[1], self.hidden_dim)
        self.rnn = TransformerDecoder(out_shape, self.hidden_dim)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=prm['lr'], weight_decay=1e-4)
        
    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.empty(0, device=device), torch.empty(0, device=device)
        
    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criterion = self.criterion.to(self.device)
        
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encoder(images)
            logits, _ = self.rnn(inputs, memory)
            
            loss = self.criterion(logits.reshape(-1, self.hidden_dim), targets.reshape(-1))
            
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
            
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        
        if captions is not None:
            caps = captions[:,0,:].to(self.device) if captions.ndim == 3 else captions.to(self.device)
            inputs = caps[:, :-1]
            logits, hidden_state = self.rnn(inputs, memory)
            return logits, hidden_state
        else:
            raise NotImplementedError()

def supported_hyperparameters() -> set:
    return {'lr', 'momentum'}