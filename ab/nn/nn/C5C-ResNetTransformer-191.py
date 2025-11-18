import torch.nn as nn
import torch
from typing import Tuple, Dict, Optional

class Net(nn.Module):
    def __init__(self, in_shape: Tuple, out_shape: Tuple, prm: Dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        
        # Encoder
        self.encoder = self._build_encoder(in_shape=in_shape, hidden_dim=640)
        
        # Decoder
        self.decoder = self._build_decoder(vocab_size=out_shape[0], hidden_size=640)
        
        # Projection layer for memory conditioning
        self.condition_layer = nn.Linear(640, 640)
        
    def _build_encoder(self, in_shape: Tuple, hidden_dim: int) -> nn.Module:
        # Define a custom CNN encoder
        encoder = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        return encoder
    
    def _build_decoder(self, vocab_size: int, hidden_size: int) -> nn.Module:
        # Define a Transformer decoder
        decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True),
            num_layers=2
        )
        decoder_embedding = nn.Embedding(vocab_size, hidden_size)
        return nn.TransformerDecoder(
            decoder_embedding, 
            decoder,
            batch_first=True
        )
    
    def train_setup(self, optimizer: torch.optim, lr: float, momentum: float) -> None:
        # Set learning rate and momentum
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            if 'momentum' in param_group:
                param_group['momentum'] = momentum
    
    def learn(self, captions: torch.Tensor, images: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Process images through encoder
        memory = self.encoder(images)
        memory = memory.squeeze(-1).squeeze(-1)  # [B, 640]
        memory = self.condition_layer(memory)   # [B, 640]
        memory = memory.unsqueeze(1)           # [B, 1, 640]
        
        # Process captions through decoder
        decoder = self.decoder
        tgt = captions[:, :-1]                  # [B, T-1]
        memory = memory.to(self.device)
        tgt = tgt.to(self.device)
        
        # Generate masks for teacher forcing
        tgt_mask = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool, device=tgt.device),
            diagonal=1
        )
        tgt_mask = torch.where(tgt_mask, -torch.inf, 0)
        
        # Forward pass through decoder
        output = decoder(
            tgt,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        # Project output to vocabulary space
        logits = self.decoder.output_proj(output)  # Assuming output_proj is defined in the decoder
        
        return logits
    
    def forward(self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shape assertions
        assert x.dim() == 4, "Input must be 4D (batch, channels, height, width)"
        assert x.size(1) == self.in_shape[1], "Input channels must match"
        assert x.size(2) == x.size(3), "Input must be square"
        
        # Process through encoder
        memory = self.encoder(x)
        memory = memory.squeeze(-1).squeeze(-1)  # [B, 640]
        memory = self.condition_layer(memory)   # [B, 640]
        memory = memory.unsqueeze(1)           # [B, 1, 640]
        
        # Project memory to hidden_size
        memory = memory.to(torch.float32)
        memory = self.decoder.condition(memory)
        
        # Return memory and hidden_state
        return memory, hidden_state

def supported_hyperparameters() -> Dict:
    return {'lr', 'momentum'}