import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # [S, B, D] for positional encoding
        pe = self.pe[:x.size(0), :]
        x = x + pe
        x = x.permute(1, 0, 2)  # back to [B, S, D]
        return self.dropout(x)

class ViTEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=16, num_layers=6, hidden_size=768, num_heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (256 // patch_size) ** 2  # Assuming 256x256 input images
        self.class_embedding = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.pos_encoding = PositionalEncoding(hidden_size, max_len=self.num_patches+1)

        # Patch embedding layer
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Flatten(start_dim=2)
        )
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Transformer encoder layers
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # Project the image into patches
        x = self.proj(x)  # [B, C, H, W] -> [B, hidden_size, num_patches] (after conv and flatten)
        B = x.size(0)
        # Add class token
        class_token = self.class_embedding.repeat(B, 1, 1)
        x = torch.cat([class_token, x], dim=1)  # [B, num_patches+1, hidden_size]
        # Add positional encoding
        x = self.pos_encoding(x)
        # Pass through transformer layers
        for block in self.blocks:
            x = block(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=6, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.pos_encoding = PositionalEncoding(hidden_size, max_len=50)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Decoder layers
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, memory):
        # tgt: [B, T]
        # memory: [B, S, hidden_size]
        tgt = tgt.to(torch.long)
        tgt = self.embedding(tgt)
        tgt = self.pos_encoding(tgt)
        memory = memory.permute(0, 2, 1)  # [B, S, hidden_size] -> [B, hidden_size, S]
        tgt = tgt.permute(0, 2, 1)  # [B, T, hidden_size] -> [B, hidden_size, T]
        out = self.blocks(tgt, memory)
        out = out.permute(0, 2, 1)  # back to [B, T, hidden_size]
        return self.fc_out(out)

class DecoderWrapper(nn.Module):
    def __init__(self, base_decoder):
        super().__init__()
        self.base_decoder = base_decoder

    def forward(self, inputs, hidden_state, memory):
        # inputs: [B, T]
        # memory: [B, S, hidden_size]
        return self.base_decoder(inputs, memory), hidden_state

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.encoder = ViTEncoder(in_shape[1])  # Now using in_shape[1] for channels
        self.decoder = TransformerDecoder(out_shape)
        self.rnn = DecoderWrapper(self.decoder)

    def supported_hyperparameters():
    return {'lr','momentum'}


    def train_setup(self, lr, momentum):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(momentum, 0.999))
        self.scheduler = None  # Placeholder for scheduler

    def learn(self, images, captions=None, hidden_state=None):
        images = images.to(self.device)
        if captions is not None:
            captions = captions.to(self.device)
            memory = self.encoder(images)
            tgt = captions[:, :-1]
            output = self.rnn(tgt, hidden_state, memory)
            return output
        else:
            # Beam search implementation would go here
            return None

    def forward(self, images, captions=None, hidden_state=None):
        images = images.to(self.device)
        memory = self.encoder(images)
        if captions is not None:
            captions = captions.to(self.device)
            tgt = captions[:, :-1]
            output = self.rnn(tgt, hidden_state, memory)
            return output, memory
        else:
            # Return initial hidden state for beam search
            return None, None

# Example usage (not part of the model)
if __name__ == "__main__":
    # Hyperparameters
    in_channels = 3
    out_vocab_size = 5000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = Net(in_shape=(in_channels, 256, 256), out_shape=out_vocab_size, prm=None, device=device)
    model = model.to(device)
    
    # Example input
    images = torch.randn(8, in_channels, 256, 256).to(device)
    captions = torch.randint(0, out_vocab_size, (8, 15)).to(device)
    
    # Forward pass
    output, memory = model(images, captions)
    print("Output shape:", output.shape)  # Should be [8, 14, 5000]
    print("Memory shape:", memory.shape)  # Should be [8, 257, 768]