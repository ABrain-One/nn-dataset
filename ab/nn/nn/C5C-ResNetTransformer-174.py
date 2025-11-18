

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: Tuple, out_shape: Tuple, prm: Dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape[0])
        self.hidden_dim = int(prm.get('hidden_dim', 768))
        # Encoder CNN backbone (ResNet-inspired) producing memory features [B, S, H]
        self.encoder = self.build_encoder(in_shape[1])
        
        # Decoder using LSTM
        self.rnn = self.build_decoder(self.vocab_size, self.hidden_dim)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        
        # Extract vocabulary size from params if needed
        vocab_size_val = prm.get('vocab_size')
        if vocab_size_val is not None:
            self.vocab_size = int(vocab_size_val)

    def build_encoder(self, in_channels: int) -> nn.Module:
        """Builds the encoder backbone similar to ResNet but adapted for caption generation."""
        encoder = nn.Sequential(
            # Stem
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Stage 1
            self._make_res_block(64, 64, 2, stride=1, downsample=None),
            
            # Stage 2
            self._make_res_block(64, 128, 2, stride=2, downsample=nn.AvgPool2d(kernel_size=2)),
            
            # Stage 3
            self._make_res_block(128, 256, 2, stride=2),
            
            # Stage 4
            self._make_res_block(256, 512, 2, stride=2)
        )
        return encoder

    def _make_res_block(self, in_c: int, out_c: int, blocks: int, stride: int, downsample: Optional[nn.Module]=None) -> nn.Sequential:
        """Helper to create a residual block group."""
        layers = []
        for i in range(blocks):
            stride_current = stride if i == 0 else 1
            downsample_current = downsample if i == 0 else None
            layer = self._residual_block(in_c, out_c, stride_current, downsample_current)
            in_c = out_c
            layers.append(layer)
        return nn.Sequential(*layers)

    def _residual_block(self, in_c: int, out_c: int, stride: int, downsample: Optional[nn.Module]=None) -> nn.Sequential:
        """Single residual block."""
        identity = nn.Identity()
        if downsample is not None:
            identity = downsample
            
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        if identity is not None:
            layers.extend([identity, nn.ReLU(inplace=True)])
        return nn.Sequential(*layers)

    def build_decoder(self, vocab_size: int, hidden_size: int) -> nn.Module:
        """Builds an LSTM decoder conditioned on encoder features."""
        return nn.LSTM(
            input_size=hidden_size,      # Size of encoder features
            hidden_size=hidden_size,     # Must match or exceed 640
            num_layers=2,                # Good balance between capacity and speed
            batch_first=True,            # Required by API
            dropout=0.2
        )
    
    def init_zero_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes hidden states for training with teacher forcing."""
        num_layers = 2
        return (
            torch.zeros(num_layers, batch_size, self.hidden_dim).to(device),
            torch.zeros(num_layers, batch_size, self.hidden_dim).to(device)
        )

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = self.criterion.to(self.device)

    def learn(self, train_data) -> None:
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            # Handle captions with varying lengths
            caps_indices = captions[:, :, 0].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps_indices[:, :-1]  # [B, T-1]
            targets = caps_indices[:, 1:]   # [B, T-1]
            
            memory = self.encoder(images)  # Shape [B, 1, hidden_dim] after global pooling
            
            # Pass inputs to LSTM decoder
            output, _ = self.rnn(inputs, None)  # Use initial hidden state from init_zero_hidden
            
            # Apply dropout to prevent overfitting
            output = F.dropout(output, p=0.2, training=True)
            
            # Project to vocabulary space
            logits = output @ self.embeddings.weight  # Simple alternative to CRNN's fully_connected
            
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
            
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        images = images