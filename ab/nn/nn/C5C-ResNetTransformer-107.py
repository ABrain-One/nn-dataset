import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        pe = torch.zeros(1, max_len, d_model)
        torch.sinh_(pe, position * (math.pi / max_len) * (1 / d_model**0.5))
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0), :, :])

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device
        
        # Encoder
        self.encoder = nn.Identity()  # TODO: Replace with custom encoder
        self.encoder = MyEncoder(in_shape=in_shape, hidden_dim=768, patch_size=32, device=device)
        
        # Decoder
        self.decoder = nn.Identity()  # TODO: Replace with custom decoder
        self.decoder = MyTransformerDecoder(vocab_size=out_shape[0], hidden_dim=768, num_layers=6, device=device)
        
        # Cross entropy loss
        self.criterion = nn.CrossEntropyLoss()

    def train_setup(self, optimizer, scheduler, **prm):
        # Set up learning rate scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

    def learn(self, images, captions=None, **prm):
        # Convert images to device
        images = images.to(self.device)
        if captions is not None:
            captions = captions.to(self.device)
            
        # Forward pass
        memory = self.encoder(images)
        logits, hidden_state = self.decoder(captions, memory)
        
        # Calculate loss
        loss = self.criterion(logits, captions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def forward(self, images, captions=None, hidden_state=None):
        # Convert images to device
        images = images.to(self.device)
        
        # Get memory from encoder
        memory = self.encoder(images)
        
        # If captions are provided, use them for teacher forcing
        if captions is not None:
            captions = captions.to(self.device)
            # Teacher forcing: use all but last caption as input
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            # Forward through decoder
            logits, hidden_state = self.decoder(inputs, memory)
            # Return logits and hidden_state
            return logits, hidden_state
        
        # Else, raise error
        raise NotImplementedError

class MyEncoder(nn.Module):
    def __init__(self, in_shape, hidden_dim=768, patch_size=32, num_layers=6, num_heads=8, dropout=0.1, device=None):
        super().__init__()
        self.in_shape = in_shape
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        
        # Convolutional stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(hidden_dim)
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, batch_first=True)
        
        # Projection layer
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, images):
        # Process images through stem
        x = self.stem(images)
        
        # Flatten spatial dimensions
        b, c, h, w = x.shape
        s = h * w
        x = x.flatten(start_dim=2).permute(0, 2, 1)  # [B, S, C]
        
        # Apply transformer
        x = self.transformer(x)
        
        # Project back to hidden_dim
        x = self.proj(x)
        
        # Rearrange to [B, S, hidden_dim]
        return x.permute(0, 2, 1)

class MyTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768, num_layers=6, num_heads=8, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, device=device)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, batch_first=True)
        
        # Final projection
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, inputs, memory):
        # inputs: [B, T] (batch_first=True)
        # memory: [B, S, hidden_dim]
        
        # Embed inputs
        embedded = self.embedding(inputs)  # [B, T, hidden_dim]
        
        # Apply positional encoding
        embedded = self.pos_encoding(embedded)
        
        # Apply transformer decoder
        seq_len = inputs.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).to(memory.device)
        out = self.transformer(embedded, memory, tgt_mask=tgt_mask)
        
        # Final projection
        logits = self.fc_out(out)
        
        # Return hidden_state as the last output representation
        hidden_state = out[:, -1, :]  # [B, hidden_dim]
        
        return logits, hidden_state

def supported_hyperparameters():
    return {'lr','momentum'}