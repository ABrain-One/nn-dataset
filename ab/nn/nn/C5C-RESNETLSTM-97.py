import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, List, Optional, Tuple


def supported_hyperparameters():
    return {'lr','momentum'}



class ResidualBlockWithSqueezeExcitation(nn.Module):
    """Modified Residual block with Squeeze-and-Excite module"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, se_ratio: float = 0.2):
        super().__init__()
        
        # Standard convolution paths
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Add Squeeze-and-Excite
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False) if se_ratio else nn.Identity(),
            nn.BatchNorm2d(out_channels) if se_ratio else nn.Identity(),
        )
        
        # Shortcut path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Squeeze-and-Excite activation if enabled
        self.se = None
        if se_ratio:
            mid_channels = max(int(out_channels * se_ratio), 32)  # Minimum 32
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, mid_channels, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, 1, sigmoid=True)
            )
            
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        x = self.main_path(x)
        
        # Apply SE activation if present
        if self.se:
            x = identity + self.se(F.relu(x))
        else:
            x = identity + x
            
        return F.relu(x)


class DynamicConvolution(nn.Module):
    """Dynamic filter convolution inspired by CondConv concept"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = False):
        super().__init__()
        self.num_filters = 4  # Fixed small number for simplicity
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias)
            for _ in range(self.num_filters)
        ])
        self.routing = nn.Parameter(torch.ones(1, in_channels, 1), requires_grad=True)
        
    def forward(self, x: Tensor) -> Tensor:
        routing = F.sigmoid(self.routing)
        routes = routing.expand(-1, x.size(1), -1)  # Match channels
        
        total_output = sum(conv(x) * route for conv, route in zip(self.convs, torch.unbind(routes, dim=1)))
        return total_output


class ImageCaptioningTransformerDecoder(nn.Module):
    """Simple Transformer decoder with learned positional encoding"""
    
    def __init__(self, d_model: int, nhead: int, num_layers: int = 1, dropout: float = 0.1, batch_first=True):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.batch_first = batch_first
        
        self.position_encoder = nn.Parameter(torch.rand(1, d_model).uniform_(-0.1, 0.1), requires_grad=True)
        
        self.decoder = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=batch_first
        )
        
        self.layers = nn.ModuleList([self.decoder.clone() for _ in range(num_layers)])
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Linear(d_model, d_model, bias=False)
        
        # Final projection to vocabulary
        self.fc_out = nn.Linear(d_model, 1)
        
    def forward(self, tgt: Tensor, src: Optional[Tensor] = None) -> Tuple[Tensor, None]:
        """src: [memory_key, memory_value, ...] in [B, num_mem, d_model] format"""
        if src is None:
            src = torch.zeros(tgt.size(0), self.nhead, self.d_model//self.nhead).to(self.device)
        
        # Learnable positional encoding (simple version)
        seq_len = tgt.size(1)
        pos_encoding = torch.cumsum(torch.arange(seq_len), dim=0).unsqueeze(-1).float()
        pos_encoding = (pos_encoding * self.position_encoder)[:, :, :self.d_model]
        pos_encoding = pos_encoding.sin() * 0.1  # Simple sinusoidal pattern
        
        embedded = self.embedding(tgt) + pos_encoding
        encoded = embedded
        
        # Run through transformer layers
        for l_idx, layer in enumerate(self.layers):
            encoded = layer(encoded, src, batch_first=self.batch_first)
            
        # Final output projection (scalar prediction per token)
        logits = self.fc_out(encoded)
        return logits.squeeze(-1), None
    
    def init_mask(self) -> None:
        """Not used here, just for compatibility"""
        pass


class CaptioningNet(nn.Module):
    """Main captioning model combining encoder and decoder with SE or Dynamic convolutions"""
    
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        
        # Parse dimensions
        in_channels = in_shape[1][1]  # Assume CHW format
        image_size = in_shape[1][2]
        self.vocab_size = out_shape[0][0]
        
        # Hyperparameter overrides
        self.hidden_size = int(prm.get('hidden_size', 768))  # Default to large dimension
        self.dropout_rate = float(prm.get('dropout', 0.1))
        self.attention_type = prm.get('attention', 'default')
        
        # Choose between SE or Dynamic convolution with probability proportional to their capabilities
        se_ratio = 0.2  # Ratio parameter for SE module
        dynamic = False  # Enable dynamic convolutions only if available computation resources
        
        if self.hidden_size < 640:
            self.hidden_size = 768  # Enforce minimum dimension
            
        # Ensure nhead divides hidden_size (Transformer recommendation)
        self.nheads = min(int(prm.get('heads', self.hidden_size//16)), 
                          self.hidden_size//32 if self.hidden_size % 32 == 0 else 8)
        if self.nheads > self.hidden_size // self.nheads:
            self.nheads = self.hidden_size // self.nheads
            
        # Encoder architecture with configurable bottleneck
        self.cnn = nn.Sequential(
            # Initial conv layers (fixed for fair comparison)
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            
            # Main bottleneck block
            ResidualBlockWithSqueezeExcitation(
                in_channels=64, 
                out_channels=self.hidden_size, 
                stride=2, 
                se_ratio=se_ratio if self.attention_type=='SE' else None
            ),
            
            # Final conv block to prepare for ViT-like operation
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        # Decoder setup
        self.transformer_decoder = ImageCaptioningTransformerDecoder(
            d_model=self.hidden_size,
            nhead=self.nheads,
            num_layers=int(prm.get('layers', 1)),
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Projection layers for input features
        self.input_projection = nn.Linear(3*self.hidden_size, self.hidden_size)
        
        # Layer normalizations for decoder
        for layer in self.transformer_decoder.layers:
            if hasattr(layer, 'norm'):
                layer.norm = nn.LayerNorm([self.hidden_size, layer.ffn.layers[0].dim])
                
    def forward(self, images: Tensor, captions: Optional[Any] = None, hidden_state: Optional[Any] = None) -> Tuple[Tensor, Any]:
        batch_size = images.size(0)
        
        # Get visual features from encoder
        features = self.cnn(images)
        # Expected shape: [B, hidden_size, 1, 1] but squeeze to 1D for compatibility
        features = features.permute(0,2,3,1)  # NHWC -> BNHC
        features = features.view(batch_size, -1, self.hidden_size)
        features = F.dropout(F.relu(self.input_projection(features)), p=self.dropout_rate)
        
        # Decoder operates in batch-first mode by default
        max_len = 20
        if captions is None:
            # Greedy decoding mode
            captions_out = []
            
            # Start with SOS token (assuming hardcoded index 1)
            sos_idx = 1
            prev_caption = torch.tensor(sos_idx, dtype=torch.long).expand(batch_size).to(self.device)
            
            for i in range(max_len):
                decoder_input = prev_caption[:,None]  # Add time dimension
                
                # Pass through transformer decoder
                logits, _, = self.transformer_decoder(decoder_input, features)
                
                # Convert logits to next token probabilities
                probs = F.softmax(logits, dim=-1)
                next_tokens = probs.argmax(dim=-1)
                
                captions_out.append(next_tokens)
                prev_caption = next_tokens
                
            return torch.stack(captions_out, dim=1), prev_caption
            
        else:
            # Teacher forcing mode
            return self.transformer_decoder(captions, features)
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999)
        )

        
    def learn(self, train_data):
        """Assumes train_data yields tuples of (images, captions)"""
        self.train()
        total_steps = 0
        running_loss = 0.0
        
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            images = images.float()
            
            # Forward pass
            logits, _ = self(images, captions)
            
            # Calculate loss
            loss = self.criteria(logits, captions)
            
            # Backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            total_steps += 1
            
            if total_steps % 32 == 0:
                avg_loss = running_loss / total_steps
                print(f"Batch {total_steps}: Loss = {avg_loss:.4f}")
                
                # Reset for next reporting period
                running_loss = 0.0


# Example instantiation (commented out)
if __name__ == "__main__":
    # Test shapes
    batch_size = 4
    img_size = 224
    channels = 3
    vocab_size = 5000
    
    # Generate dummy data
    dummy_images = torch.randn(batch_size, channels, img_size, img_size)
    dummy_captions = torch.randint(0, vocab_size, (batch_size, 30))
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Instantiate model with hyperparameters (example)
    model = CaptioningNet(
        in_shape=((channels, img_size, img_size),),
        out_shape=(vocab_size,),
        prm={'lr': 0.0001, 'momentum': 0.95, 'hidden_size': 768, 'heads': 8, 'dropout': 0.1},
        device=device
    )
    
    # Print stats
    print(model)
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Input shape: {dummy_images.shape}")
    print(f"Output shape: {model(dummy_images, captions=None)[0].shape}")