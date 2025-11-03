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
        self.vocab_size = int(out_shape)
        # Encoder configuration parameters
        base_channels = 64
        
        # Encoder: ResNet-like feature extractor
        layers = []
        
        # Initial convolution layer
        layers.append(nn.Conv2d(in_shape[1], base_channels, kernel_size=7, stride=2, padding=3, bias=False))
        layers.append(nn.BatchNorm2d(base_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Add more convolutional blocks to achieve sufficient hidden dimension
        # Stage 1: Multiple blocks to meet H>=640 requirement
        for i in range(3):
            layers.append(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(base_channels))
            layers.append(nn.ReLU(inplace=True))
        
        # Stage 2
        base_channels *= 2
        layers.append(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(base_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for i in range(4):
            layers.append(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(base_channels))
            layers.append(nn.ReLU(inplace=True))
        
        # Stage 3 (final block to push hidden dimension)
        base_channels *= 2
        layers.append(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(base_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Final extraction to get memory features [B, S, H]
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.encoder = nn.Sequential(*layers)
        
        # Determine input size and calculate total sequence length S
        test_image = torch.randn(1, in_shape[1], in_shape[2], in_shape[3]).to(self.device)
        encoded_img = self.encoder(test_image).flatten(1, 3).permute(0, 2, 1)  # Shape: [B, H, S]
        self.seq_length = encoded_img.shape[2]
        self.hidden_dim = base_channels
        
        # Adjust hidden dimension to meet minimum requirement (>640)
        if self.hidden_dim < 640:
            self.hidden_dim = 640
            
        # Decoder: Transformer decoder with cross attention
        # Initialize parameters
        num_heads = min(8, self.hidden_dim // 64)  # Minimum of 8 heads or based on dimension
        self.attention_dropout = 0.1
        self.ffn_dim = self.hidden_dim * 4
        
        # Create decoder layers
        decoder_layers = []
        for _ in range(3):
            # Self-attention layer for decoder
            decoder_layers.append(nn.MultiheadAttention(self.hidden_dim, num_heads, dropout=self.attention_dropout, batch_first=True))
            
            # Feed-forward network after self-attention
            layers_ffn = nn.Sequential(
                nn.Linear(self.hidden_dim, self.ffn_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.ffn_dim, self.hidden_dim)
            )
            decoder_layers.append(layers_ffn)
            
            # Skip connections and dropout
            decoder_layers.append(nn.Dropout(0.1))
            # Residual connection
            decoder_layers.append(lambda x: x + F.linear(F.relu(self.encoder_proj(x)), self.encoder_proj.weight / 32.0))  # Approximate residual connection
        
        # Build the encoder projection for memory (used later in FFN components)
        self.encoder_proj = nn.Linear(self.seq_length, self.hidden_dim)  # Project temporal context to hidden dimension
        
        # Actual Transformer decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=num_heads,
                dim_feedforward=self.ffn_dim,
                batch_first=True,
                dropout=self.attention_dropout
            ),
            num_layers=len(decoder_layers)//3  # Number of full decoder blocks
        )
        
        # Add the processed decoder layers
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Final prediction layer
        self.output_layer = nn.Linear(self.hidden_dim, self.vocab_size)
        
        # Ensure correct initialization to prevent vanishing/exploding gradients
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

    def init_zero_hidden(self, batch: int, device: torch.device):
        # Returns zero-initialization for the Transformer decoder's hidden state
        return torch.zeros(1, batch, self.hidden_dim).to(device)

    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1).to(self.device)

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            memory = self.encode_images(images)
            logits, hidden_state = self.decode_sequence(inputs, memory)
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def encode_images(self, images):
        """Extract memory features from images using the encoder backbone"""
        batch_size = images.shape[0]
        encoded = self.encoder(images)  # Shape: [B, H, S] after feature extraction
        encoded = encoded.flatten(start_dim=1, end_dim=2).permute(0, 2, 1)  # Shape: [B, S, H]
        return encoded

    def decode_sequence(self, inputs, memory):
        """
        Decode the caption sequence using Teacher Forcing
        
        Args:
            inputs: caption input tokens [B, T] (if using teacher forcing, will be [:,-1] for next token)
            memory: encoder memory [B, S, H]
            
        Returns:
            (logits, hidden_state): prediction probabilities and final hidden state
        """
        batch_size, seq_len = inputs.shape
        emb_size = memory.shape[-1] if len(memory.shape) > 2 else memory.shape[-1]
        
        # Process the input sequence through decoder embedding
        embeddings = inputs.unsqueeze(-1).expand(-1, -1, emb_size)  # Shape: [B, T, H]
        
        # Run through the decoder layers
        hidden = self.init_zero_hidden(batch_size, self.device)
        for layer in self.decoder:
            if isinstance(layer, nn.MultiheadAttention):
                # Perform cross attention with memory
                attn_output, _ = layer(embeddings.transpose(1,2), 
                                      key_padding_mask=~torch.isnan(memory).any(dim=-1).float())
                embeddings = attn_output.transpose(1,2)
            elif hasattr(layer, 'linear') and isinstance(layer, nn.Sequential):
                # Process through the feed-forward layers
                embeddings = layer(embeddings)
                
        # Apply final linear transformation to predictions
        logits = self.output_layer(embeddings)  # Shape: [B, T, vocab_size]
        
        # Return the predicted logits and the final hidden state (even though Transformer)
        # provides no traditional hidden state, maintaining API consistency
        return logits, hidden
    
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encode_images(images)
        if captions is not None:
            captions_inputs = captions[:,0,:-1]  # All tokens except last
            return self.decode_sequence(captions_inputs, memory)
        else:
            raise NotImplementedError()