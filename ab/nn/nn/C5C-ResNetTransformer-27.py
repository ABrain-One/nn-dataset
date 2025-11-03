import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        
        # Encoder backbone
        self.encoder = CNNEncoder(in_shape, out_shape, device)
        
        # Decoder
        self.decoder = TransformerDecoder(out_shape, in_shape)
        
        # Initialize weights
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name or 'bias' in name:
                        nn.init.uniform_(param, -0.05, 0.05)
    
    def train_setup(self, prm: dict):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=prm['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
    def learn(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        captions = captions.to(self.device)
        
        # Get memory features from encoder
        memory = self.encoder(images)
        
        # Forward through decoder with teacher forcing
        logits, _ = self.decoder(captions, None, memory)
        
        # Flatten the logits and captions for loss calculation
        logits = logits.reshape(-1, self.out_shape)
        captions = captions.reshape(-1)
        
        # Calculate loss
        loss = F.cross_entropy(logits, captions)
        
        return loss
    
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, 
                hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        images = images.to(self.device)
        
        # Get memory features from encoder
        memory = self.encoder(images)
        
        # Forward through decoder
        logits, hidden_state = self.decoder(captions, hidden_state, memory)
        
        return logits, hidden_state

class CNNEncoder(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: int, device: torch.device):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        
        # Define the encoder layers
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(512, out_shape)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        
        # Forward through the encoder layers
        x = self.relu1(self.bn1(self.conv1(images)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        
        # Global average pooling
        x = self.pool(x)
        
        # Project to the output shape
        x = self.projection(x.squeeze(-1).squeeze(-1))
        
        # Reshape to [B, 1, H] where H is the output shape
        return x.unsqueeze(1)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8, batch_first=True),
            num_layers=6
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, seq: torch.Tensor, hidden_state: Optional[torch.Tensor] = None, 
                memory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        seq = seq.to(self.device)
        memory = memory.to(self.device)
        
        # Embed the sequence
        embedded = self.embed(seq)
        embedded = self.pos_encoder(embedded)
        
        # Expand memory to match the sequence length
        memory = memory.expand(-1, embedded.size(1), -1)
        
        # Forward through transformer decoder
        output = self.transformer(embedded, memory)
        
        # Project to vocabulary space
        logits = self.fc_out(output)
        
        return logits, None

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(d_model).float() * -torch.log(torch.tensor(10000.0)) / d_model)
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0:d_model//2] = torch.sin(position * div_term[0:d_model//2])
        pe[:, 0, d_model//2:d_model] = torch.cos(position * div_term[d_model//2:d_model])
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pe = self.pe[:seq_len]
        x = x + pe
        return self.dropout(x)

def supported_hyperparameters():
    return {'lr', 'momentum'}