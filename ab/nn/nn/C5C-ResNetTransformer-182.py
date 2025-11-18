

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def supported_hyperparameters():
    return {'lr','momentum'}


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.register_buffer('pe', self._generate_pe(max_len, d_model))

    def _generate_pe(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        div_term = torch.exp(torch.arange(0, d_model // 2, dtype=torch.float) *
                             (-math.log(10000.0) / (d_model // 2)))
        
        # First for even indices, then odd
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        sin = torch.sin(pos / 10000.0 * div_term).expand(-1, d_model // 2)
        cos = torch.cos(pos / 10000.0 * div_term).expand(-1, d_model // 2)
        pe = torch.cat([sin, cos], dim=1)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.pe[:x.size(1), :].unsqueeze(0)
        return x + pe

class Net(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int, int], out_shape: int, prm: Dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        
        # Extract input properties
        self.channels = in_shape[1]
        self.image_size = in_shape[2]
        
        # Initialize model hyperparameters
        self.vocab_size = out_shape
        self.hidden_dim = 1024  # ≥640
        self.num_heads = 8     # Must divide hidden_dim, 1024 ÷ 8 = 128
        self.num_layers = 6
        
        # Build encoder backbone (modified AlexNet-inspired)
        self.encoder = nn.Sequential(
            # Block 1: Initial convolution reducing channels
            BasicConv2d(self.channels, 64, kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Block 2: Intermediate processing layers
            BasicConv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BasicConv2d(128, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 512, kernel_size=3, stride=1, padding=1),
            BasicConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            
            # Block 3: Final feature extraction layers
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * 512, self.hidden_dim)
        )
        
        # Set up decoder
        self.decoder = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=1024,
            batch_first=True
        )
        self.transformer_dec = nn.TransformerDecoder(
            self.decoder,
            num_layers=self.num_layers,
            norm=None,
            batch_first=True
        )
        
        # Word embedding layer
        self.embedding_layer = nn.Embedding(
            self.vocab_size,
            self.hidden_dim,
            padding_idx=0
        )
        
        # Projection layers
        self.projection_layer = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(0.1)
        )
        
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)
        
        # Loss function initialization
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize decoder hidden states"""
        return (
            torch.zeros(batch, self.hidden_dim, device=device),
            torch.zeros(batch, self.hidden_dim, device=device)
        )

    def train_setup(self, prm: Dict) -> None:
        """Setup for training"""
        self.to(self.device)
        
        # Optimizer configuration
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(beta1, 0.999)
        )

    def learn(self, train_data: Iterator) -> None:
        """Main training loop"""
        self.train()
        
        for images, captions in train_data:
            # Transfer data to device
            images = images.to(self.device, dtype=torch.float32)
            targets = captions[:, 1:].contiguous()
            inputs = captions[:, :-1].contiguous()
            
            # Forward pass
            memory = self.encoder(images)  # Shape: [B, 1, H] (H=1024)
            decoded, _ = self.transformer_dec(
                tgt=inputs,
                memory=memory
            )
            
            # Calculate loss
            logits = self.fc_out(decoded)
            loss = self.criterion(
                logits.view(-1, self.vocab_size),
                targets.view(-1)
            )
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                max_norm=3.0
            )
            self.optimizer.step()

    def forward(self, 
                 images: torch.Tensor, 
                 captions: Optional[torch.Tensor] = None, 
                 hidden_state=None) -> Union[Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass with teacher forcing"""
        if captions is not None:
            # Training phase with teacher forcing
            images_flat = images.view(-1, *images.shape[2:])
            memory = self.encoder(images_flat)  # Shape: [B, 1, 1024]
            
            # Mask for decoder
            seq_len = captions.size(1)
            dec_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=captions.device),
                diagonal=1
# --- auto-closed by AlterCaptionNN ---
)
