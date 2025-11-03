import math


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        
        # Extract dimensions and ensure proper handling
        self.in_channels = int(in_shape[1])
        self.hidden_dim = 640  # Meeting the minimum requirement
        self.vocab_size = int(out_shape[0])
        
        # Create encoder with customizable layers
        layers = [
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.GlobalAveragePooling2d(),
            nn.Linear(512, self.hidden_dim)
        ]
        self.encoder = nn.Sequential(*layers)
        
        # Adjust the last linear layer to achieve at least 640 hidden dimension if needed
        if self.hidden_dim < 640:
            self.project_encoder = nn.Linear(self.hidden_dim, 640)
        else:
            self.project_encoder = nn.Identity()
            
        # Decoder selection: Using nn.TransformerDecoder for this implementation
        # Parameters adjusted to maintain divisibility requirements
        embedding_dim = self.hidden_dim
        num_heads = 8
        if embedding_dim % num_heads != 0:
            # Adjustments to make dimensions compatible
            divisor = num_heads
            remainder = embedding_dim % divisor
            embedding_dim -= remainder
            # Set appropriate hidden dimensions if needed
            print(f"Warning: Changed embedding dimension to {embedding_dim} to maintain compatibility with num_heads={num_heads}")
            
        self.transformer_dec = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.transformer_dec, num_layers=6)
        
        # Embedding layer mapping integers to real-valued vectors
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        
        # Projection layer from decoder outputs to vocabulary space
        self.projection = nn.Linear(embedding_dim, self.vocab_size, bias=False)
        
        # Initialize attention masks properly
        self.pos_encoder = None
        self.pos_decoder = None
        
        # Loss function and optimizer initialization
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        
    def init_zero_hidden(self, batch: int, device: torch.device):
        """Initialize empty hidden states for decoder processing."""
        return (torch.empty(0), torch.empty(0)), torch.empty(0).to(device)  # Modified for transformer decoder needs
    
    def train_setup(self, prm: dict):
        """Configure model for training using provided parameters."""
        self.to(self.device)
        self.criterion = self.criterion.to(self.device)
        
        # Optimizer settings from parameters dictionary
        lr = float(prm.get('lr', 1e-3))
        momentum = float(prm.get('momentum', 0.9))
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(momentum, 0.999))
    
    def learn(self, train_data):
        """Train model on batches from data loader."""
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            
            # Process captions appropriately based on their dimensions
            if captions.ndim == 3:
                caps = captions[:, 0, :].long().to(self.device)
            else:
                caps = captions.long().to(self.device)
                
            inputs = caps[:, :-1]      # Teacher forcing: everything except last prediction
            targets = caps[:, 1:]        # Teacher forcing: ground truth without BOS token
            
            # Encode visual features
            memory = self.encode_visual(images)
            
            # Decode text predictions conditioned on memory features
            logits, hidden_state = self.decode_text(inputs, memory)
            
            # Calculate loss using teacher-forced targets
            loss = self.criterion(logits.reshape(-1, self.vocab_size), 
                                 targets.reshape(-1))
                                 
            # Update model parameters
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
    
    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        """Process images through encoder to extract meaningful features."""
        # Raw image encoding
        raw_feats = self.encoder[:-(2+self.hidden_dim!=640)].modules()  # All layers except final projections
        
        # Intermediate activations for richer feature extraction
        intermediate_feats = []
        x = images
        for layer in self.encoder[:-(2+self.hidden_dim!=640)]:
            x = layer(x)
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d):
                intermediate_feats.append(F.relu(x))
        
        # Combine all feature maps into a hierarchical representation
        concatenated_feats = torch.cat([layer(x) for layer in self.encoder], dim=1)
        
        # Projection into unified hidden dimension
        pooled_feats = self.project_encoder(concatenated_feats.flatten(start_dim=2)).transpose(1, 2)
        return pooled_feats
    
    def decode_text(self, inputs: torch.Tensor, memory: torch.Tensor) -> tuple:
        """Generate text predictions conditioned on encoded visual features."""
        # Apply positional encoding for temporal awareness
        seq_length = inputs.size(1)
        position_ids = torch.arange(0, seq_length, dtype=torch.long).unsqueeze(0).to(self.device)
        embedded = self.embedding(inputs) + self.embed_pos(position_ids)
        
        # Compute attention masks for auto-regressive modeling
        tgt_mask = torch.triu(torch.ones_like(embedded), diagonal=1)
        tgt_mask[tgt_mask.bool()] = float('-inf')
        
        # Transformer decoding with teacher forcing
        attn_output = self.decoder(embedded, memory, tgt_mask=tgt_mask)
        
        # Final projection onto vocabulary space
        logits = self.projection(attn_output.transpose(1, 2))
        
        return logits, attn_output  # Returning full attention output matrix
        
    def embed_pos(self, position: torch.Tensor) -> torch.Tensor:
        """Simple positional embedding for transformer decoder."""
        if self.pos_encoder is None:
            self.pos_encoder = nn.Embedding(position.size()[1]+1, self.hidden_dim).to(self.device)
            self.pos_encoder.weight.data.copy_(self.create_sinusoidal_embedding(position.size()[1]+1, self.hidden_dim))
        return self.pos_encoder(position)
        
    @staticmethod
    def create_sinusoidal_embedding(sequence_length: int, embedding_dim: int) -> torch.Tensor:
        """Create standard sinusoidal positional encoding for transformers."""
        # Source: standard implementation from fairseq repo
        # More precisely, adapted from FastCopy's documentation
        positions = torch.arange(0, sequence_length).unsqueeze(1)
        scale = -math.log(10000.0) / embedding_dim
        div_term = torch.exp(scale * torch.arange(0, embedding_dim, 2))
        
        sin_embeddings