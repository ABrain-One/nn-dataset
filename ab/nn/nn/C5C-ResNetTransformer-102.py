import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape)
        in_channels = int(in_shape[1])
        self.hidden_dim = 768  # Must be ≥640

        # Encoder: CNN backbone with attention features
        self.encoder = nn.Sequential(
            # Stem
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Blocks
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Additional stages to ensure sufficient features
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.hidden_dim),
            nn.Tanh()
        )

        # Decoder: Transformer-based with cross-attention
        d_model = self.hidden_dim
        num_layers = 3
        num_heads = 4  # Should divide d_model, hence d_model needs to be divisible by num_heads

        # Verify hidden_dim is divisible by num_heads
        assert d_model % num_heads == 0, "Hidden dimension must be divisible by number of attention heads"
        
        self.decoder = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=2048, batch_first=True)
        self.embeddings = nn.Embedding(self.vocab_size, d_model)
        
        self.position_encoding = PositionalEncoding(d_model, max_len=50)  # Assuming reasonable max length
        
        self.fc_out = nn.Linear(d_model, self.vocab_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=float(prm.get('dropout', 0.1)))
        
    def forward_encoder(self, images: torch.Tensor) -> torch.Tensor:
        """Helper function for shared interface"""
        return self.encoder(images)

    def init_zero_hidden(self, batch: int, device: torch.device):
        """Initialize encoder-decoder integration"""
        # No special hidden state management needed for transformer
        return None, None

    def train_setup(self, prm: dict):
        """Optimization setup"""
        self.to(self.device)
        self.embeddings.to(self.device)
        self.fc_out.to(self.device)
        self.decoder.to(self.device)
        
        # Standard optimization parameters
        params = {
            'lr': float(prm.get('lr', 1e-3)),
            'weight_decay': float(prm.get('weight_decay', 1e-4)),
            'betas': (float(prm.get('momentum', 0.95)), 0.999),
            'eps': 1e-6
        }
        
        # Use AdamW with standard settings
        self.optimizer = torch.optim.AdamW(self.parameters(), **params)

    def learn(self, train_data):
        """Standard training loop"""
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:, 0, :].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.forward_encoder(images)  # [B, 1, 768]
            logits, _ = self.decoder(inputs, memory)  # [B, L-1, D]
            
            loss = F.cross_entropy(logits.transpose(1, 2), targets.flatten(), ignore_index=0)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def batch_first_forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None):
        """Main inference/prediction interface"""
        images = images.to(self.device, dtype=torch.float32)
        memory = self.forward_encoder(images)  # [B, 1, 768]
        
        if captions is not None:
            caps = captions[:, 0, :].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            logits = self.decode_step(inputs, None, memory)  # [B, L-1, D]
            
            return logits, None
        else:
            raise NotImplementedError()

    def decode_step(self, inputs: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor]], memory: torch.Tensor) -> Tuple[torch.Tensor]:
        """Step-by-step decoding logic"""
        # Apply embeddings and positional encoding
        encoded = self.embeddings(inputs)  # [B, L, D]
        encoded = self.position_encoding(encoded)  # [B, L, D]
        
        # Transformer decoding
        decoded = self.decoder(encoded, memory)  # [B, L, D]
        
        # Final prediction
        logits = self.fc_out(decoded)  # [B, L, V]
        
        return logits, None

class PositionalEncoding(nn.Module):
    """Learnable positional encoding based on sine/cosine"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        inv_freq = self.div_term.unsqueeze(0) * torch.ones((pos.shape[0], 1))
        self.encoding[:, 0::2] = torch.sin(pos * inv_freq)
        self.encoding[:, 1::2] = torch.cos(pos * inv_freq)
        self.encoding = self.encoding.float().unsqueeze(0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:x.size(0), :]

# Example usage demonstrates the core functionality
if __name__ == "__main__":
    # Test input shapes
    in_shape = (3, 224, 224)  # Batch x Channels x Height x Width
    out_shape = (5000,)  # Vectors of dimension 5000 for demonstration
    
    # Dummy parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_net = Net(in_shape, out_shape, {'lr': 1e-3}, device)
    
    # Sample input
    fake_images = torch.rand((8, 3, 224, 224), device=device)
    memory = test_net.forward_encoder(fake_images)
    print(f"Memory features shape: {memory.shape}")  # Expect [8, 1, 768]
    
    # Sample captions
    fake_caps = torch.randint(0, 5000, (8, 15), device=device)
    
    # Forward pass
    test_net.train_setup({'lr': 1e-3})
    logits, hidden = test_net.learn([(fake_images, fake_caps)])
    print(f"Output logits shape: {logits.shape}")  # Expect [8, 14, 5000]