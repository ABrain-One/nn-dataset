import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        pe = torch.zeros(max_len, 1, d_model)
        pe[0, 0, 0] = 1
        for pos in range(1, max_len):
            pe[pos, 0, 0] = 0
            for i in range(1, d_model//2+1):
                # 2i or 2i+1
                period = 2 * i - 1
                # sin(pos * 2π / period)
                pe[pos, 0, 2*i-1] = torch.sin(-0.5 * pos / (period * 1000**(2*i-1/d_model)))
                pe[pos, 0, 2*i] = torch.cos(-0.5 * pos / (period * 1000**(2*i-1/d_model)))
        self.pe = pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be of shape [seq_len, batch_size, d_model]
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :, :].squeeze(1)
        return self.dropout(x)

class CNN_Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2048),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.projector1 = nn.Linear(64, out_channels)
        self.projector2 = nn.Linear(512, out_channels)
        self.projector3 = nn.Linear(2048, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W]
        x1 = self.stage1(x)
        x1 = F.adaptive_avg_pool2d(x1, (1,1)).squeeze(2).squeeze(2)
        x1 = self.projector1(x1).unsqueeze(1)
        
        x2 = self.stage2(x)
        x2 = F.adaptive_avg_pool2d(x2, (1,1)).squeeze(2).squeeze(2)
        x2 = self.projector2(x2).unsqueeze(1)
        
        x3 = self.stage3(x)
        x3 = F.adaptive_avg_pool2d(x3, (1,1)).squeeze(2).squeeze(2)
        x3 = self.projector3(x3).unsqueeze(1)
        
        memory = torch.cat([x1, x2, x3], dim=1)
        return memory

class LSTM_Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [B, T-1]
        # memory shape: [B, S, H]
        # hidden_state shape: [B, hidden_size]
        
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)  # [B, 1, d_model]
        embedded = torch.cat([embedded, memory], dim=1)  # [B, S+1, d_model]
        
        out, hidden = self.lstm(embedded, hidden_state)
        logits = self.fc(out.squeeze(1))
        return logits, hidden

class Net(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], out_shape: Tuple[int, ...], **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.vocab_size = out_shape[0]
        
        # Define the encoder
        self.encoder = CNN_Encoder(in_shape[1], 768)
        
        # Define the decoder
        self.decoder = LSTM_Decoder(self.vocab_size, 768, 768)
        
        # Initialize the hidden state
        self.hidden_size = 768
        self.hidden = None

    def train_setup(self, **kwargs):
        # Set up the training parameters
        self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs.get('lr', 0.001))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **kwargs)

    def learn(self, images: torch.Tensor, captions: torch.Tensor, memory: torch.Tensor = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute memory from images if not provided
        if memory is None:
            memory = self.encoder(images)
        
        # Initialize hidden state if not provided
        if self.hidden is None:
            self.hidden = self.decoder.lstm.hidden_size
        
        # Forward pass through decoder
        logits, hidden = self.decoder(captions, memory, self.hidden)
        self.hidden = hidden
        
        # Calculate loss
        loss = F.cross_entropy(logits, captions.argmax(1), ignore_index=0)
        return logits, hidden

    def forward(self, images: torch.Tensor, captions: torch.Tensor = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # If captions are provided, use learn method
        if captions is not None:
            return self.learn(images, captions, **kwargs)
        
        # Otherwise, generate captions
        # This is a placeholder for the generation part
        # We'll return a dummy output for now
        return torch.zeros((images.size(0), 1, self.vocab_size)), torch.zeros((images.size(0), self.hidden_size))

def supported_hyperparameters() -> Dict[str, Any]:
    return {'lr', 'momentum'}

# --- auto-closed by AlterCaptionNN ---