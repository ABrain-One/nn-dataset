import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional

class Net(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], out_shape: Tuple[int, ...], **prm):
        super().__init__()
        self.image_size = in_shape[2]  # assuming square image
        self.hidden_dim = 768  # >=640

        # TODO: Replace self.encoder with custom encoder producing memory tensor [B, S, H] where H >=640
        self.encoder = VisionTransformerEncoder(
            in_channels=in_shape[1],
            image_size=self.image_size,
            patch_size=16,
            num_layers=12,
            num_heads=8,
            hidden_dim=self.hidden_dim,
            mlp_dim=self.hidden_dim*4,
            dropout=0.1,
            attention_dropout=0.1)

        # TODO: Replace self.rnn with custom decoder implementing forward(inputs, None, memory) -> (logits, hidden_state)
        self.rnn = TransformerDecoder(
            hidden_dim=self.hidden_dim,
            vocab_size=out_shape[0],
            num_layers=6,
            num_heads=8)

    def train_setup(self, **prm):
        # Set up the model for training
        pass

    def learn(self, inputs: Tensor, targets: Tensor, features: Tensor, **prm) -> Tensor:
        # Teacher forcing training loop
        logits, hidden_state = self.rnn(inputs, None, features)
        # Calculate loss
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        loss = nn.functional.cross_entropy(logits, targets, **prm)
        return loss

    def forward(self, x: Tensor, captions: Optional[Tensor] = None, **prm) -> Tuple[Tensor, Tensor]:
        # If captions are provided, use teacher forcing
        if captions is not None:
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            features = self.encoder(x)
            logits, hidden_state = self.rnn(inputs, None, features)
            return logits, hidden_state
        else:
            # For inference, we need to handle differently
            raise NotImplementedError("Inference mode is not implemented in this version")

class VisionTransformerEncoder(nn.Module):
    def __init__(self, in_channels: int, image_size: int, patch_size: int, num_layers: int, num_heads: int, hidden_dim: int, 
                 mlp_dim: int, dropout: float, attention_dropout: float):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim).normal_(std=0.02))
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=attention_dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = torch.permute(x, (0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
        x = torch.flatten(x, start_dim=1)   # [B, H*W*C]
        x = torch.reshape(x, (x.shape[0], x.shape[1] // self.proj.in_channels * self.proj.in_channels, -1))
        x = torch.permute(x, (0, 2, 1))      # [B, num_patches, hidden_dim]
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = torch.permute(x, (1, 0, 2))      # [num_patches, B, hidden_dim]
        x = self.transformer(x)
        x = torch.permute(x, (1, 0, 2))      # [B, num_patches, hidden_dim]
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, num_layers: int, num_heads: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tuple[Tensor, Tensor]:
        embedded = self.embed_tokens(tgt)
        embedded = self.pos_encoder(embedded)
        out = self.transformer_decoder(tgt=embedded, memory=memory)
        logits = self.fc_out(out)
        return logits, out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(torch.arange(max_len).float() * div_term)
        pe[:, 1::2] = torch.cos(torch.arange(max_len).float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

def supported_hyperparameters():
    return {'lr','momentum'}