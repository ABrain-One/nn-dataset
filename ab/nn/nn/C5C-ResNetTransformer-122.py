import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        
        # Encoder configuration
        in_channels = int(in_shape[1])
        height, width = int(in_shape[2]), int(in_shape[3])
        self.hidden_dim = 768  # H ≥ 640
        
        # Create encoder with ViT-like structure
        self.encoder = nn.Sequential(
            # Stem layer
            nn.Conv2d(in_channels, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Downsampling stages
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Final layer before classification head
            nn.Conv2d(256, self.hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
        )
        
        # Ensure correct output shape [B, S, H]
        # Calculate actual S based on input dimensions and operations
        output_size = (height // 2, width // 2)  # Reduced by 2 factors (first conv and maxpool not applied explicitly here)
        output_tensor = F.relu(torch.rand(1, in_channels, height, width))  # Random tensor for demonstration
        after_stem = F.relu(output_tensor.clone())
        after_downsample = F.relu(after_stem)
        after_final = self.encoder[0:2](output_tensor)  # First two layers (stem)
        final_spatial = (after_final.shape[-2], after_final.shape[-1])
        self.seq_length = final_spatial[0] * final_spatial[1]
        
        # Adjusting encoder if necessary
        if final_spatial[0] * final_spatial[1] > 200 and self.hidden_dim < 640:
            print("Increasing hidden dimension due to large sequence length")
            self.hidden_dim = 1024
        
        self.seq_length = final_spatial[0] * final_spatial[1]
        assert self.seq_length > 0 and self.hidden_dim >= 640, "Invalid hidden dimension or sequence length"
        
        # Decoder configuration
        self.vocab_size = out_shape[0]  # Assuming out_shape is (vocab_size,)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        num_layers = 6
        num_heads = 8
        dim_feedforward = 2048
        
        self.decoder = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            enable_nested_tensor=True
        )
        self.transformer_dec = nn.TransformerDecoder(
            self.decoder,
            num_layers=num_layers,
            enable_nested_tensor=True
        )
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)
        
        # Standard settings for decoder
        self.pos_encoder = nn.Parameter(torch.zeros(1, self.seq_length, self.hidden_dim))
        self.register_parameter('pos_encoder', self.pos_encoder)
        nn.init.xavier_uniform_(self.pos_encoder)
        
        # Initialize position encodings properly
        self.initialize_positional_encodings()

    def initialize_positional_encodings(self):
        """Initialize positional encoding weights"""
        pos_embed = torch.zeros(self.seq_length, self.hidden_dim).requires_grad_()
        nn.init.trunc_normal_(pos_embed, mean=0.0, std=0.02, a=-1.0, b=1.0)
        self.pos_encoder.data.copy_(pos_embed.t())

    def train_setup(self, prm: Dict):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1).to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=float(prm.get('lr', 1e-3)))
        
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            # Determine caption dimensions
            if captions.ndim == 3:
                targets = captions[:,1:].contiguous()
                inputs = captions[:, :-1]
            else:
                targets = captions[:,1:].contiguous()
                inputs = captions[:, :-1]
                
            memory = self.encoder(images)
            # Expand memory to [B, S, H] ensuring proper batch handling
            expanded_memory = memory.unsqueeze(1) if memory.dim() == 3 else memory
                
            # Ensure memory and inputs are correctly shaped
            if expanded_memory.size(1) != self.seq_length or inputs.size(1) != self.seq_length - 1:
                continue
                
            embedded_inputs = self.embedding(inputs)
            embedded_inputs = embedded_inputs + self.pos_encoder[:inputs.size(0)]
            logits = self.transformer_dec(embedded_inputs, expanded_memory)
            logits = self.fc_out(logits)
            
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.empty((batch, self.seq_length, self.hidden_dim), device=device)

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        
        # Handle decoder inputs if captions are provided
        if captions is not None:
            # Determine caption dimensions
            if captions.ndim == 3:
                sequence_length = captions.size(1) - 1
                batch_size = captions.size(0)
                targets = captions[:,1:]
                inputs = captions[:, :-1]
            else:
                sequence_length = captions.size(1) - 1
                batch_size = captions.size(0)
                targets = captions[:,1:]
                inputs = captions[:, :-1]
                
            # Expand memory if needed
            if memory.dim() == 2:
                memory = memory.unsqueeze(0)
                
            # Get final hidden state if it wasn't provided
            if hidden_state is None:
                hidden_state = self.init_zero_hidden(batch_size, self.device)
                
            # Process inputs through transformer
            embedded_inputs = self.embedding(inputs)
            # Add positional encoding to inputs
            if hidden_state.dim() == 2:
                # Adjust if hidden_state isn't properly shaped
                extended_state = hidden_state.unsqueeze(1).expand(batch_size, self.seq_length, self.hidden_dim)
            else:
                extended_state = hidden_state
                
            # Combine embeddings with positional info
            embedded_inputs = embedded_inputs + extended_state
                
            # Decode and project to output space
            logits = self.transformer_dec(embedded_inputs, memory)
            logits = self.fc_out(logits)
            
            # Check dimensions
            assert logits.shape == torch.Size([batch_size, sequence_length, self.vocab_size])
            return logits, extended_state
            
        else:
            raise NotImplementedError("Decoding without captions is not yet implemented")