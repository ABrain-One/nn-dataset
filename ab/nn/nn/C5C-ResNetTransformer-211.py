import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

def supported_hyperparameters():
    return {'lr','momentum'}


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Base convolution layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        
        # SE-block after each convolution level
        self.se1 = self._squeeze_excitation(32, 16)
        self.se2 = self._squeeze_excitation(64, 8)
        self.se3 = self._squeeze_excitation(128, 4)
        
        # Projection to desired hidden dimension
        self.proj = nn.Linear(128, hidden_dim)

    def _squeeze_excitation(self, input_channels, squeeze_factor):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, input_channels//squeeze_factor, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(input_channels//squeeze_factor, input_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsample path with attention
        out1 = self.pool1(F.relu(self.bn1(self.conv1(x))))
        out1 = F.relu(self.se1(out1))
        
        out2 = self.pool2(F.relu(self.bn2(self.conv2(out1))))
        out2 = F.relu(self.se2(out2))
        
        out3 = F.relu(self.bn3(self.conv3(out2)))
        out3 = F.relu(self.se3(out3))
        
        # Global pooling and projection
        pooled = F.adaptive_avg_pool2d(out3, (1, 1))
        proj = self.proj(pooled)
        return proj

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, vocab_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.embed_dim = hidden_dim
        
        self.tok_embeddings = nn.Embedding(vocab_size, self.embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, 
                                               nhead=num_heads,
                                               dim_feedforward=vocab_size,
                                               batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.lm_head = nn.Linear(self.embed_dim, vocab_size)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        # tgt: target sequence [B, T-1]
        # memory: encoder output [B, S, D] (D=H=hidden_dim)
        
        seq_len = tgt.size(1)
        if tgt_mask is None:
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(tgt.device)
        
        # Embed and apply positional encoding
        embedded = self.tok_embeddings(tgt)  # [B, T-1, D]
        out = self.decoder(embedded, memory, tgt_mask=tgt_mask)
        logits = self.lm_head(out)  # [B, T-1, V]
        return logits

class Net(nn.Module):
    def __init__(self, in_shape: Tuple, out_shape: int, prm: Dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape
        
        # Configuration from hyperparameters
        self.hidden_dim = int(prm.get('hidden_dim', 768))
        self.num_heads = int(prm.get('num_heads', 8))
        
        # Encoder initialization
        if self.hidden_dim < 640:
            raise ValueError(f"Hidden dimension {self.hidden_dim} must be at least 640")
        self.encoder = Encoder(in_channels=in_shape[1], hidden_dim=self.hidden_dim)
        
        # Decoder initialization
        self.decoder = TransformerDecoder(hidden_dim=self.hidden_dim, 
                                       num_heads=self.num_heads,
                                       vocab_size=self.vocab_size)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = None

    def init_zero_hidden(self, batch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros((batch, self.hidden_dim), device=self.device),
                torch.zeros((batch, self.hidden_dim), device=self.device))

    def train_setup(self, prm: Dict):
        self.to(self.device)
        self.criterion = self.criterion.to(self.device)
        
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encoder(images)  # [B, 1, H] -> [B, 1, hidden_dim]
            assert inputs.ndim == 2, "Expected 2D inputs for decoder"
            logits, _ = self.decoder(inputs, memory)  # [B, T-1, V]
            assert logits.shape == (inputs.size(0), inputs.size(1), self.vocab_size)
            
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor]=None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)  # Shape: [B, 1, H]
        
        if captions is not None:
            if captions.ndim == 2:
                captions = captions.to(self.device)
            else:
                captions = captions[:,0,:].to(self.device)
                
            inputs = captions[:, :-1]
            logits = self.decoder(inputs, memory)  # Shape: [B, T-1, V]
            return logits, hidden_state
        else:
            # Decoder inference logic goes here
            raise NotImplementedError("Inference without captions not implemented")

    def decode_with_teacher_forcing(self, images, captions):
        """Helper function for teacher forcing during training"""
        memory = self.encoder(images)
        return self.forward(images, captions, None)