import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import Optional, Tuple

class ConvStemConfig:
    def __init__(self, out_channels: int, kernel_size: int, stride: int, norm_layer: nn.Module, activation_layer: nn.Module):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer

class ConvStem(nn.Module):
    def __init__(self, config: ConvStemConfig):
        super().__init__()
        self.config = config
        self.conv = nn.Conv2d(config.out_channels, config.out_channels, kernel_size=config.kernel_size, stride=config.stride)
        self.norm = config.norm_layer(config.out_channels)
        self.activation = config.activation_layer(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: float, attention_dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x)
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=x.device)
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        attn_output = nn.functional.dropout(attn_output, p=self.dropout, training=self.training)
        out = x + attn_output
        out = self.norm2(out)
        ff_output = self.ff(out)
        ff_output = nn.functional.dropout(ff_output, p=self.dropout, training=self.training)
        out = out + ff_output
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe[:, 0]  # [max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return x

class Net(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], out_shape: Tuple[int,], prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape[0]
        self.hidden_dim = 768  # Ensure H>=640

        image_channels = in_shape[1] if len(in_shape)==3 else 3
        img_size = in_shape[1]  # Assuming square image

        # Use the closest split as per the second example
        def get_closest_split(n: int, divisor: int) -> int:
            while n % divisor != 0:
                divisor //= 2
            return divisor

        self.patch_size = get_closest_split(img_size, prm.get('patch_size', 32))
        n_patches = (img_size // self.patch_size) ** 2

        # Encoder
        self.stem = ConvStem(
            config=ConvStemConfig(
                out_channels=image_channels,
                kernel_size=7,
                stride=2,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU
            )
        )
        self.proj = nn.Conv2d(image_channels, self.hidden_dim, kernel_size=1, stride=1, bias=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, self.hidden_dim))

        encoder_layer = EncoderBlock(
            num_heads=8,
            hidden_dim=self.hidden_dim,
            mlp_dim=3072,
            dropout=0.1,
            attention_dropout=0.0
        )
        self.encoder = nn.Sequential(
            *[
                encoder_layer for _ in range(prm.get('num_encoder_layers', 12))
            ]
        )

        # Decoder
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            encoder_layer=encoder_layer,
            num_layers=prm.get('num_decoder_layers', 3),
            batch_first=True
        )
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)

    def init_zero_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns empty tensors for hidden_state and cell_state
        return (torch.zeros(1, 1, self.hidden_dim, device=self.device), torch.zeros(1, 1, self.hidden_dim, device=self.device))

    def train_setup(self, lr: float, momentum: float) -> None:
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

    def learn(self, images: torch.Tensor, captions: torch.Tensor) -> None:
        # Train on batches
        self.train()
        for i in range(captions.shape[1]):
            if i == 0:
                continue  # Skip first step for teacher forcing
            cap_input = captions[:, i, :]
            cap_target = captions[:, i+1, :] if i+1 < captions.shape[1] else captions[:, i, :]
            # Forward pass
            memory = self.encode(images)
            cap_input_emb = self.token_embedding(cap_input)
            cap_input_emb = cap_input_emb.permute(1, 0, 2)  # [B, T, D]
            mask = self.generate_square_subsequent_mask(cap_input_emb.size(1), self.device).to(self.device)
            out_emb = self.transformer_decoder(tgt=cap_input_emb, memory=memory, tgt_mask=mask)
            logits = self.fc_out(out_emb)
            # Loss calculation
            loss = nn.functional.cross_entropy(logits.view(-1, self.vocab_size), cap_target.view(-1))
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # If captions are provided, use teacher forcing
        if captions is not None:
            if captions.ndim == 3:
                captions = captions.squeeze(1)  # Flatten the first dimension
            # Get input and target tokens
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            # Encode images
            memory = self.encode(images)
            # Embed captions
            embedded = self.token_embedding(inputs)
            embedded = embedded.permute(1, 0, 2)  # [B, T, D]
            # Generate mask for decoder
            mask = self.generate_square_subsequent_mask(embedded.size(1), self.device).to(self.device)
            # Decode
            outputs = self.transformer_decoder(tgt=embedded, memory=memory, tgt_mask=mask)
            # Project to vocabulary
            logits = self.fc_out(outputs)
            # Return logits and hidden_state (here hidden_state is None)
            return logits, None
        else:
            raise NotImplementedError("Beam search generation is not implemented in this forward pass")

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        # Pass through the encoder
        x = self.stem(images)
        x = self.proj(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        # Add class token
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1)
        # Add positional encoding
        x = x + self.pos_embed
        # Pass through encoder layers
        for layer in self.encoder:
            x = layer(x)
        return x

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        # Generate a square mask for the sequence length
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).type(torch.bool)
        return mask == False

    def supported_hyperparameters():
        return {'lr', 'momentum'}