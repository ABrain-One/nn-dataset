import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any

class Net(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], out_shape: int, 
                 prm: Dict[str, Any], device: torch.device) -> None:
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) > 1 else 3
        self.vocab_size = out_shape[0]
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size
        # Backward-compat local aliases (old LLM patterns)
        self.vocab_size = self.vocab_size
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size
        self.in_channels = self.in_channels
        
        # Encoder
        self.cnn = GridFeatureEncoder(in_shape, self.vocab_size)
        
        # Decoder
        self.rnn = TransformerCaptioningDecoder(self.vocab_size, self.vocab_size)
        
        # Projection layer
        self.proj = nn.Linear(self.vocab_size, self.vocab_size)
        
    def train_setup(self, prm: Dict[str, Any]) -> None:
        pass
        
    def learn(self, train_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        total_loss = 0.0
        total_correct = 0
        total = 0
        
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Append SOS token
            sos = torch.full((images.size(0), 1), self.sos_idx, device=self.device)
            inputs = torch.cat([sos, captions[:, :-1]], dim=1)
            targets = captions[:, 1:]
            
            # Forward pass
            logits, _ = self.forward(images, inputs)
            
            # Flatten for loss calculation
            logits_flat = logits.reshape(-1, self.vocab_size)
            targets_flat = targets.reshape(-1)
            
            # CrossEntropyLoss
            loss = F.cross_entropy(logits_flat, targets_flat)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits_flat, dim=1)
            correct = (predicted == targets_flat).sum().item()
            total_correct += correct
            total += targets_flat.size(0)
            
        avg_loss = total_loss / len(train_data)
        accuracy = total_correct / total
        
        return avg_loss, accuracy
        
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, 
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Any]:
        # If captions are provided, use teacher forcing
        if captions is not None:
            # Append SOS token
            sos = torch.full((images.size(0), 1), self.sos_idx, device=images.device)
            captions = torch.cat([sos, captions[:, :-1]], dim=1)
            
            # Get encoder features
            encoder_features = self.cnn(images)
            
            # Decoder forward pass
            outputs_sequence, final_hidden_state = self.rnn(captions, None, features=encoder_features)
            
            # Project outputs
            outputs_projected = self.proj(outputs_sequence)
            
            return outputs_projected, final_hidden_state
        
        # Otherwise, generate captions
        else:
            # Start with SOS token and generate until eos or max_len
            captions = torch.full((images.size(0), 1), self.sos_idx, device=images.device)
            outputs = []
            
            # Decoder forward pass
            outputs_sequence, final_hidden_state = self.rnn(captions, None, features=self.cnn(images))
            
            # Project outputs
            outputs_projected = self.proj(outputs_sequence)
            
            # Convert to tokens
            _, predicted_tokens = torch.max(outputs_projected, dim=2)
            captions = torch.cat([captions, predicted_tokens], dim=1)
            
            return captions, final_hidden_state

class GridFeatureEncoder(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], output_dim: int) -> None:
        super().__init__()
        self.in_channels = in_shape[1] if len(in_shape) > 1 else 3
        self.num_patches = 196
        
        # Stage1
        self.stage1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stage2
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Stage3
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Final projection
        self.token_proj = nn.Linear(256, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        x = self.token_proj(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pe = self.pe[:seq_len]
        x = x + pe
        return self.dropout(x)

class TransformerCaptioningDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.transformer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=2048)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(batch, self.embed_dim, device=device), torch.zeros(batch, self.embed_dim, device=device)
        
    def forward(self, inputs: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_len = inputs.size(1)
        batch_size = inputs.size(0)
        device = inputs.device
        
        # Embedding and positional encoding
        embedded = self.embeddings(inputs)
        embedded = self.pos_encoder(embedded)
        
        # Transformer decoding
        output = self.transformer(embedded, None, features)
        
        # Project to vocabulary
        logits = self.fc_out(output)
        
        # Final hidden state
        final_hidden_state = (output, None)
        
        return logits, final_hidden_state

def supported_hyperparameters():
    return {'lr','momentum'}


# Note: The following lines are required for the code to run properly
# We need to import math for the PositionalEncoding
import math

# Define SOS token index
Net.sos_idx = 0