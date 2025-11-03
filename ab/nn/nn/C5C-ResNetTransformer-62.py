import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

class BagNetUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, expansion: int = 1) -> None:
        super().__init__()
        self.expansion = expansion
        self.padding = padding
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], out_shape: Tuple[int, ...], prm: Dict[str, Any], device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.vocab_size = out_shape[0]
        self.image_size = in_shape[2]
        self.patch_size = prm.get('patch_size', 0.2)
        self.num_heads = prm.get('num_heads', 8)
        self.hidden_dim = 768
        
        # Build the stem
        stem_configs = [
            BagNetUnit(3, 64, 3, 2, 1),
            BagNetUnit(64, 64, 3, 1, 1),
            BagNetUnit(64, 128, 3, 2, 1),
            BagNetUnit(128, 128, 3, 1, 1),
            BagNetUnit(128, 256, 3, 2, 1),
            BagNetUnit(256, 256, 3, 1, 1),
            BagNetUnit(256, 512, 3, 2, 1),
            BagNetUnit(512, 512, 3, 1, 1),
            BagNetUnit(512, 768, 3, 2, 1)
        ]
        self.stem = nn.Sequential(*stem_configs)
        
        # Define the class token
        self.class_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        # Define the projection for the stem output
        self.conv_proj = nn.Conv2d(768, self.hidden_dim, kernel_size=1)
        
        # Define the transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads, dim_feedforward=3072, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Define the decoder embedding layer
        self.decoder_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        
        # Define the decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=self.num_heads, dim_feedforward=3072, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # Define the final classification head
        self.classifier = nn.Linear(self.hidden_dim, self.vocab_size)
    
    def train_setup(self, prm: Dict[str, Any]) -> None:
        # Set up the model for training
        self.train()
        self.encoder.dropout = prm.get('dropout', 0.1)
        self.decoder_embedding.dropout = prm.get('dropout', 0.1)
    
    def learn(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # This function is called during training with teacher forcing
        if captions is None:
            raise ValueError("Captions must be provided for teacher forcing")
        
        # Process the images to get memory features
        x = self.stem(images)
        x = self._process_input(x)
        x = torch.cat([self.class_token.expand(images.size(0), -1, -1), x], dim=1)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        
        # Project the captions to embeddings
        if captions.dim() == 3:
            captions = captions[:, :-1]
        captions = self.decoder_embedding(captions)
        captions = captions.permute(1, 0, 2)
        
        # Run the decoder
        memory = x
        tgt = captions
        output = self.decoder(tgt, memory)
        output = output.permute(0, 2, 1)
        
        # Project to vocabulary space
        logits = self.classifier(output)
        logits = logits.permute(0, 2, 1)
        
        # Return the logits and a hidden_state (dummy)
        hidden_state = torch.zeros(logits.size(0), self.hidden_dim).to(self.device)
        return logits, hidden_state
    
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # The API requires that if captions are provided, we return (logits, hidden_state)
        # Otherwise, we return the generated captions
        
        if captions is not None:
            return self.learn(images, captions)
        else:
            # For evaluation, generate captions without teacher forcing
            # This is a placeholder implementation
            x = self.stem(images)
            x = self._process_input(x)
            x = torch.cat([self.class_token.expand(images.size(0), -1, -1), x], dim=1)
            x = x.permute(0, 2, 1)
            x = self.encoder(x)
            
            # Generate initial token
            initial_input = torch.full((images.size(0), 1), 0).long().to(self.device)
            initial_input = self.decoder_embedding(initial_input)
            initial_input = initial_input.permute(1, 0, 2)
            
            # Run the decoder for one step
            memory = x
            tgt = initial_input
            output = self.decoder(tgt, memory)
            output = output.permute(0, 2, 1)
            
            # Project to vocabulary space
            logits = self.classifier(output)
            logits = logits.permute(0, 1, 2)
            
            # Return a dummy hidden_state
            hidden_state = torch.zeros(logits.size(0), self.hidden_dim).to(self.device)
            return logits, hidden_state
    
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        # This function processes the input image to get the patch embeddings
        n, c, h, w = x.shape
        torch._assert(h == self.image_size and w == self.image_size, "Image must be square and of size " + str(self.image_size))
        
        # Calculate the number of patches
        n_h = h // self.patch_size
        n_w = w // self.patch_size
        
        # Project the input to the hidden dimension
        x = self.conv_proj(x)
        
        # Reshape to [B, C, N_h, N_w] and then flatten the spatial dimensions
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        return x

def supported_hyperparameters() -> Dict[str, Any]:
    return {'lr', 'momentum'}