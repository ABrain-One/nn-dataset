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
        self.vocab_size = int(out_shape[0])
        in_channels = int(in_shape[1])
        self.hidden_dim = 640
        
        # Encoder: Modified CNN backbone with global pooling and projection to 640-dimension space
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 640)
        )
        
        # Decoder: Transformer Decoder with cross-attention
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=640, nhead=8, dim_feedforward=2048, batch_first=True),
            num_layers=6,
            norm=None  # Will apply normalization in post-processing
        )
        
        # Word embeddings
        self.embedding = nn.Embedding(self.vocab_size, 640)
        
        # Projection layer to match vocabulary size
        self.fc_proj = nn.Linear(640, self.vocab_size)
        
        # Project encoded image features to match decoder context
        self.project_encoder = nn.Sequential(
            nn.Linear(640, 640)
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.optimizer = None
        
        # Initialize parameters with Xavier uniform distribution
        self._initialize_params()
    
    def _initialize_params(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def init_zero_hidden(self, batch: int, device: torch.device):
        # No hidden state needed for transformer-based decoder
        return torch.zeros((batch, 640), device=device)
    
    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = self.criterion.to(self.device)
    
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encode_image(images)
            logits = self.decode_sequence(inputs, memory)
            
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
    
    def encode_image(self, images: torch.Tensor):
        # Shape: [B, S, H] → [B, 1, 640]
        cnn_features = self.encoder(images)
        projected_features = self.project_encoder(cnn_features)
        return projected_features
    
    def decode_sequence(self, inputs: torch.Tensor, memory: torch.Tensor):
        # [B, T-1] → [B, T-1, 640] via embedding
        embedded_inputs = self.embedding(inputs)
        
        # Decoding with transformer cross-attention (teacher forcing)
        attn_mask = torch.zeros_like(embedded_inputs)
        if inputs.size(1) > 1:
            # Autoregressive mask for future prediction
            seq_length = inputs.size(1)
            lower_triangle = torch.tril(attn_mask.new_full(size=(seq_length, seq_length), fill_value=-float('Inf')))
            attn_mask.scatter_(1, torch.arange(seq_length), lower_triangle)
        
        # Transformer decoder expects keys/values matching encoder output format
        # Cross-attend between decoder queries and encoder key/value content
        memory = memory.transpose(0, 1)  # [B, 640] → [640, B] for decoder processing
        
        # Run transformer decoder with attention masking
        tgt_mask = attn_mask
        memory_mask = None
        
        memory_key_padding_mask = None
        memory_value_padding_mask = None
        
        decoder_output = self.decoder(
            embedded_inputs, 
            memory, 
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_value_padding_mask=memory_value_padding_mask
        )
        
        # Project to vocabulary space
        logits = self.fc_proj(decoder_output)
        return logits, decoder_output
    
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encode_image(images)
        
        if captions is not None:
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            logits = self.decode_sequence(inputs, memory)
            return logits
        else:
            # During inference, generate captions using teacher forcing would be incorrect unless you're using beam search with history
            # Generating from scratch isn't handled by teacher forcing alone, but the question asks specifically for teacher forcing during training
            raise NotImplementedError("For teacher forcing inference, please provide captions.")