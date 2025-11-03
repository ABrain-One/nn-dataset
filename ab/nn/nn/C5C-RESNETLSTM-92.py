import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class SEBlock(nn.Module):
    def __init__(self, channel: int, ratio: int = 4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x)
        y = self.excite(y)
        return x * y.expand_as(x)

class PatchEmbedding(nn.Module):
    """Convert images to patch tokens"""
    def __init__(self, in_channels: int, patch_size: int, embedding_dim: int):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):
    """Modified Transformer decoder layer"""
    def __init__(self, embedding_dim: int, num_heads: int):
        super(DecoderBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self attention
        q, k, v = x, x, x
        x = self.norm1(q)
        x = self.self_attention(x, q, k)[0]
        x = self.dropout(x)
        
        # Cross attention if memory provided
        if memory is not None:
            x = self.norm2(x)
            x_cross, _ = self.cross_attention(x, memory, memory)
            x = x + x_cross
            
        # Feed forward
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        return x

class CustomTransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_layers: int, num_heads: int):
        super(CustomTransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PatchEmbedding(3, 16, embedding_dim)  # Simple learnable positional encoding
        
        # Decoder layers
        decoder_blocks = []
        for _ in range(num_layers):
            decoder_blocks.append(DecoderBlock(embedding_dim, num_heads))
        self.decoder_blocks = nn.Sequential(*decoder_blocks)
        
        # Final projection
        self.projection = nn.Linear(embedding_dim, vocab_size)
        
    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(batch, self.embedding.num_embeddings, self.embedding.embedding_dim, 
                           device=device), 
                torch.zeros(batch, self.embedding.num_embeddings, self.embedding.embedding_dim, 
                           device=device))
    
    def forward(self, inputs: torch.Tensor, memory: Optional[torch.Tensor] = None, 
                hidden_state: Optional[Tuple[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if memory is None and hidden_state is None:
            raise ValueError("Either memory or hidden_state must be provided")
            
        # Apply embedding and positional encoding
        embedded = self.embedding(inputs)
        embedded_pos = self.pos_encoding(inputs).transpose(0, 1)
        embedded = embedded + embedded_pos
        
        # Process through decoder layers
        for block in self.decoder_blocks:
            embedded = block(embedded, memory)
            
        # Final projection
        logits = self.projection(embedded.transpose(0, 1))
        
        # Return last hidden state of decoder blocks (avg of all layers)
        if memory is None:
            h_states = [block.norm2.weight.detach().clone() for block in self.decoder_blocks]
            avg_state = torch.stack(h_states).mean(dim=0)
        else:
            avg_state = memory[-1][:,-1,:]
                
        return logits, avg_state

def supported_hyperparameters():
    return {'lr','momentum'}



class Net(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], out_shape: Tuple[int, ...], 
                 prm: Dict[str, Any], device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        
        # Extract dimensions
        height, width = in_shape[1], in_shape[2]
        embedding_dim = prm.get('hidden_size', 640)
        num_layers = prm.get('num_layers', 1)
        num_heads = prm.get('num_heads', 8)
        image_size = min(height, width)
        patch_size = prm.get('patch_size', 4)
        
        # Define encoder components
        self.patch_embedding = PatchEmbedding(in_shape[1], patch_size, embedding_dim)
        self.backbone = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim*2, 3, padding=1),
            nn.LeakyReLU(0.1),
            SEBlock(embedding_dim*2),
            nn.Conv2d(embedding_dim*2, embedding_dim, 3, padding=1)
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Define decoder
        self.decoder = CustomTransformerDecoder(
            out_shape[0], embedding_dim, num_layers, num_heads
        )
        
        # Classifier layers
        self.classifier_dropout = nn.Dropout(0.2)
        self.classifier_final = nn.Linear(embedding_dim, out_shape[0])
        
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999)
        )


    
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Extract features with patch embedding and backbone
            patches = self.patch_embedding(images)
            features = self.backbone(patches)
            global_feat = self.global_context(features)
            
            # Get text embeddings
            text_inputs = captions
            
            # Forward pass through decoder
            logits, _ = self.decoder(text_inputs, None)
            
            # Reshape and compute loss
            logits_flat = logits.view(-1, out_shape[0])
            targets_flat = captions.view(-1)
            
            loss = self.criteria[0](logits_flat, targets_flat)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract features with patch embedding and backbone
        patches = self.patch_embedding(images)
        features = self.backbone(patches)
        global_feat = self.global_context(features)
        
        # Get text embeddings
        text_inputs = captions
        
        # Forward pass through decoder
        logits, avg_state = self.decoder(text_inputs, None)
        
        # Reshape the logits and captions to match the output shape
        logits_flat = logits.view(-1, out_shape[0])
        targets_flat = captions.view(-1)
        
        return logits_flat, avg_state