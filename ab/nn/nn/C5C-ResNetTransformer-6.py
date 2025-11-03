import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape)
        in_channels = int(in_shape[1])
        
        # Set hidden dimension based on input constraints
        hidden_dim_value = 768  # Ensure H ≥ 640
        
        # Encoder with feature extraction capabilities
        self.encoder = nn.Sequential(
            # First convolution block
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Second convolution block
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Final layer to capture context features
            nn.Conv2d(128, hidden_dim_value, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim_value),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.hidden_dim = hidden_dim_value
        
        # Decoder with Transformer architecture
        self.transformer_dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim_value,
            nhead=min(hidden_dim_value // 8, 16),  # Balanced number of heads
            dim_feedforward=hidden_dim_value * 2,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_dec_layer,
            num_layers=6
        )
        
        # Embedding layer and final projection
        self.embedding = nn.Embedding(self.vocab_size, hidden_dim_value)
        self.fc_out = nn.Linear(hidden_dim_value, self.vocab_size)
        
        # Hyperparameters validation callback
        self.validate_dims = lambda: None
    
    def init_zero_hidden(self, batch: int, device: torch.device):
        # Initialize initial hidden state for decoder
        seq_len = 0
        if hasattr(self.transformer_decoder, '_memo'):
            seq_len = self.transformer_decoder._memo.size(1) if self.transformer_decoder._memo is not None else 0
        return torch.zeros(seq_len, batch, self.hidden_dim, device=device)

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.embedding = self.embedding.to(self.device)
        self.fc_out = self.fc_out.to(self.device)
        self.transformer_decoder = self.transformer_decoder.to(self.device)

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            memory = self.encoder(images)
            
            # Cross-attention conditioning
            memory = memory.view(memory.size(0), 1, self.hidden_dim)
            tgt = inputs
            
            # Add positional encoding to improve decoder performance
            seq_len = tgt.size(1)
            pos_encoding = torch.zeros(seq_len, self.hidden_dim).float().to(self.device)
            for pos in range(seq_len):
                pos_encoding[pos] = (
                    math.sin(pos / 10000 ** (2/self.hidden_dim)) if pos % 2 == 0 else
                    math.cos(pos / 10000 ** (2/self.hidden_dim))
                ).view(1, -1)
                
            tgt_with_pos = self.embedding(tgt) + pos_encoding[:, None, :]
            
            # Transformer decoding
            out = self.transformer_decoder(tgt_with_pos, memory)
            logits = self.fc_out(out)
            
            # Calculate loss with proper padding masking
            pad_idx = self.embedding.padding_idx if self.embedding.padding_idx is not None else 0
            loss = F.cross_entropy(logits.transpose(1, 2), targets.contiguous(), ignore_index=pad_idx)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
    
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        memory = memory.view(memory.size(0), 1, self.hidden_dim)  # Conditioning for decoder
        
        if captions is not None:
            # Teacher forcing implementation
            seq_len = captions.size(1) - 1
            input_ids = captions[:, :seq_len]
            
            # Create padded tensor
            padded_input = torch.full_like(input_ids, self.vocab_size-1)
            valid_range = (input_ids >= 0) & (input_ids < self.vocab_size)
            padded_input[valid_range] = input_ids[valid_range]
            
            # Embedding transformation
            embedded = self.embedding(padded_input)
            # Optional positional encoding addition
            # pos_encoding = torch.zeros(seq_len, self.hidden_dim).float().to(self.device)
            # for pos in range(seq_len):
            #     pos_encoding[pos] = ...
            # embedded = embedded + pos_encoding[:, None, :]
            
            # Transformer decoding
            transformed = self.transformer_decoder(
                embedded, 
                memory
            )
            
            # Final projection to vocabulary space
            logits = self.fc_out(transformed)
            
            # Reshape for appropriate metrics calculation
            logits = logits.transpose(1, 2)  # Shape [B, T, V]
            
            return logits, hidden_state if hidden_state is not None else torch.zeros_like(logits[:, :, 0])
        
        raise NotImplementedError("Inference mode generation is not implemented in this version")

def supported_hyperparameters():
    return {'lr', 'momentum'}