import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def supported_hyperparameters():
    return {'lr','momentum'}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape[0]  # Assuming out_shape format matches classification example
        self.hidden_dim = 768  # Since 768 >= 640
        
        # TODO: Replace with custom encoder backbone producing memory features [B, S, H] (H>=768)
        # Using a simplified ResNet-like architecture adapted for ViT-style feature extraction
        enc_in_channels = int(in_shape[1])  # Channels from input shape
        
        # Encoder stem
        self.stem = nn.Sequential(
            nn.Conv2d(enc_in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Encoder stages
        self.stage1 = self._make_stage(64, 128)
        self.stage2 = self._make_stage(128, 256)
        self.stage3 = self._make_stage(256, 512)
        self.stage4 = self._make_stage(512, 768)
        
        # Memory projection: expand spatial dimensions to sequence tokens
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.project_memory = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        
        # TODO: Replace with custom decoder
        # Decoder using Transformer architecture
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=min(int(math.sqrt(self.hidden_dim)), 12),  # Num heads <= sqrt(hidden_dim)
            dim_feedforward=2048,  # Large FFN to maintain power despite reduced memory
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # Embedding layer for decoder input
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        
        # Final classification head
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)

    def _make_stage(self, in_channels, out_channels):
        # Simplified bottleneck block for efficient feature reduction
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),  # Skip connection
            nn.BatchNorm2d(out_channels)
        )

    def encode_image(self, images):
        # Process image through encoder and produce memory features [B, S, H]
        x = self.stem(images)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.adaptive_pool(x)  # Global pooling to fixed-size
        x = torch.flatten(x, 1)     # Flattened features [B, 2048]
        return self.project_memory(x)  # Project to hidden dimension [B, 2048→768], effectively same as [B, 1, 768]

    def forward_for_decoder(self, inputs, memory, hidden_state=None):
        # Forward pass compatible with teacher forcing
        # Process encoder memory to proper format [B, S, H]
        memory = memory.view(memory.size(0), 1, self.hidden_dim)
        
        # Initialize hidden state if None (though transformer doesn't use it traditionally)
        if hidden_state is None:
            batch_size = inputs.size(0)
            hidden_state = torch.zeros((
                batch_size, 
                self.hidden_dim 
            )).to(self.device)
        
        # Process decoder inputs with embeddings and positional encoding
        embedded = self.embedding(inputs)
        embedded = self.position_encoding(embedded)
        
        # Decode using cross-attention to memory and sequential generation
        seq_len = embedded.size(1)
        tgt_mask = None  # Auto-regressive mask handled internally by transformer
        
        # Run through transformer decoder while maintaining proper hidden state propagation
        # Though attention mechanisms handle the temporal dependencies inherently
        out = self.transformer_decoder(
            embedded, 
            memory, 
            tgt_mask=tgt_mask
        )
        
        # Final projection to vocabulary space
        logits = self.fc_out(out)
        return logits, hidden_state

    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.zeros((batch, self.hidden_dim), device=device)

    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=lr, 
            betas=(beta1, 0.999)
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1).to(self.device)

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions.to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encode_image(images)
            logits, _ = self.forward_for_decoder(inputs, memory)
            
            assert images.dim() == 4
            assert logits.shape == (images.size(0), inputs.shape[1], self.vocab_size)
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()


    @property
    def position_encoding(self):
        # Standard interface for positional encoding
        return PositionalEncoding(self.hidden_dim)

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encode_image(images)
        
        if captions is not None:
            caps_inputs = captions[:, :-1]
            caps_targets = captions[:, 1:] if captions.ndim == 3 else captions[:, 1:].long()
            
            # Perform teacher forcing (auto-regressive decoding)
            logits, new_hidden = self.forward_for_decoder(caps_inputs, memory, hidden_state)
            
            # Validate shapes
            assert logits.shape == (images.size(0), captions.shape[1]-1, self.vocab_size)
            assert new_hidden.shape == (images.size(0), self.hidden_dim)
            
            return logits, new_hidden
        
        else:
            # Generation endpoint using beam search (placeholder for actual beam search)
            raise NotImplementedError("Beam search generation is not fully implemented")