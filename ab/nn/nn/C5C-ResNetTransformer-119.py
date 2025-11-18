import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

def supported_hyperparameters():
    return {'lr','momentum'}


class DPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=True):
        super(DPNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        if expansion:
            self.conv3 = nn.Conv2d(out_channels, out_channels//4, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels//4)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels//4, out_channels//4, bias=False),
                nn.Sigmoid()
            )
        else:
            self.conv3 = None
            self.bn3 = None
            self.se = None
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        if self.conv3 is not None:
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)
            out = out * self.se(out)
            out += identity
        else:
            out += identity
            out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super(Net, self).__init__()
        self.device = device
        
        # Encoder backbone parameters
        self.in_channels = int(in_shape[1])
        self.hidden_dim = 768  # Must be ≥640
        
        # Define the encoder with sufficient capacity
        self.encoder = nn.Sequential(
            # Initial convolution
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Downsampling path
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Expansion blocks
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=4),
        )
        
        # Calculate feature size after encoder
        test_tensor = torch.randn(1, self.in_channels, 224, 224).to(device)
        features = self.encoder(test_tensor)
        b, c, h, w = features.shape
        
        # Resize the feature map while maintaining divisibility by 2
        self.target_feature_dim = (h * w)//2**5
        
        # Project to desired hidden dimension (note: features.shape is [1, 512, h, w])
        # We reshape [B, 512, H*W] → [B, H*W, 512] then project to hidden_dim
        self.proj = nn.Linear(c * (self.target_feature_dim), self.hidden_dim)
        
        # Decoder parameters
        self.vocab_size = int(out_shape)
        self.seq_length_max = 20  # Maximum sequence length for generation
        
        # Choose decoder: Transformer with attention matching encoder dim
        num_layers = 3
        d_model = self.hidden_dim
        num_heads = self.hidden_dim // 16  # Split dim into heads
        self.decoder = nn.TransformerDecoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            batch_first=True,
            masking=False
        )
        
        self.transformer_dec = nn.TransformerDecoder(
            self.decoder,
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        # Word embedding layer
        self.embedding = nn.Embedding(self.vocab_size+1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Classifier projection to vocabulary
        self.classifier = nn.Linear(d_model, self.vocab_size)
        
        # Initialize word dictionaries (if available in params)
        self.word2idx = getattr(prm, 'word2idx', {})
        self.idx2word = getattr(prm, 'idx2word', {})
        
        # Learning related parameters
        self.learning_rate = max(float(prm.get('lr', 1e-3)), 3e-4)
        self.beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        
        # Dropout protection
        self.criterion = None
        self.optimizer = None
        self.train_mode = True
    
    def init_zero_hidden(self, batch: int, device: torch.device):
        """Initialize encoder and decoder hidden states"""
        # Encoder hidden state represents image features
        img_feat = torch.randn(batch, self.target_feature_dim, self.hidden_dim).to(device)
        proj_feat = img_feat.clone()
        
        # Decoder hidden state starts with encoder features
        decoder_hidden = img_feat.clone()
        
        return decoder_hidden
    
    def train_setup(self, prm: Dict):
        """Set up training hyperparameters and optimization"""
        self.to(self.device)
        
        # Configure optimizer with AdamW
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, 0.999)
        )
        
        # Set up loss criterion
        if hasattr(prm, 'word2idx') and prm.word2idx:
            pad_idx = prm.word2idx.get('<PAD>', 0)
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=pad_idx,
                label_smoothing=0.1
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=0,
                label_smoothing=0.1
            )
        
        # Initialize word dictionaries
        self.word2idx = getattr(prm, 'word2idx', {})
        self.idx2word = getattr(prm, 'idx2word', {})
    
    def learn(self, train_data):
        """Train on batches using teacher forcing strategy"""
        self.train()
        
        for images, captions in train_data:
            # Move data to device
            images = images.to(self.device, dtype=torch.float32)
            caps = captions.to(self.device) if captions is not None else None
            
            # Process captions (ensure proper shape for batching)
            if caps is not None and caps.ndim == 3:
                # Teacher forcing: last token is target
                inputs = caps[:, :, :-1]
                targets = caps[:, :, 1:]
            elif caps is not None:
                # Simple case: 2D captions [B, T] becomes [B, T-1]
                inputs = caps[:, :-1]
                targets = caps[:, 1:]
            else:
                # Default behavior if captions format unexpected
                continue
                
            # Encode image features
            memory = self.encode_images(images)
            
            # Decode caption tokens from inputs
            decoder_outputs = self.decode_sequence(inputs, memory)
            
            # Calculate loss and perform backward propagation
            flat_loss = self.calculate_loss(decoder_outputs, targets)
            
            if flat_loss is not None and flat_loss.requires_grad:
                self.optimizer.zero_grad()
                flat_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    max_norm=3.0
                )
                self.optimizer.step()
    
    def encode_images(self, images: torch.Tensor):
        """Process input images into contextual features"""
        # Run through encoder backbone
        x = self.encoder(images)
        
        # Permute and prepare for processing
        x = x.permute(0, 2, 1).reshape(images.size(0), self.target_feature_dim, self.hidden_dim)
        
        # Project to decoder dimension space
        x = self.proj(x)
        
        return x
    
    def decode_sequence(self, tgt_inputs: torch.Tensor, memory: torch.Tensor):
        """Generate caption sequence using encoder features"""
        # Get source and target sequences
        src = memory
        tgt = tgt_inputs
        
        # Get device information once upfront
        device = tgt_inputs.device
        
        # Add positional encoding to target inputs
        encoded_tgts = self.embedding(tgt_inputs) * math.sqrt(self.decoder.d_model)
        encoded_tgts = self.pos_encoder(encoded_tgts)
        
        # Create transformer decoder context
        decoder_context = self.init_zero_hidden(
            batch=tgt_inputs.size(0),
            device=device
        )
        
        # Determine maximum sequence length dynamically
        max_len = tgt_inputs.size(1) if tgt_inputs.dim() > 1 else self.seq_length_max
        
        # Generate sequence step-by-step
        all_logits = []
        all_hidden = []
        
        # Iterate through each generation step
        for step_idx in range(max_len):
            # Current target subsequence
            current_input = tgt[:, step_idx:step_idx+1] if tgt.dim() > 1 else tgt[:, :step_idx+1]
            
            # Query decoder for next token prediction
            decoder_output = self.transformer_dec(
                current_input,
                src,
                memory_key_padding_mask=None,
                return_intermediate=False
            )
            
            # Generate candidate distributions
            logits = self.classifier(decoder_output)
            
            # Store intermediate results if needed
            if self.training:
                # Only return immediate output in training
                return logits
            else:
                # Save predictions for later comparison
                all_logits.append(F.log_softmax(logits, dim=-1))
                all_hidden.append(decoder_output.detach())
                
        return torch.cat(all_logits, dim=1), torch.cat(all_hidden, dim=1)
    
    def calculate_loss(self, logits: torch.Tensor, targets: torch.Tensor):
        """Compute cross entropy loss considering possible sequences"""
        if targets.dim() == 3:
            # Flatten the sequence dimension for standard calculation
            targets_flat = targets.flatten()
            logits_flat = logits.flatten(start_dims=len(targets.shape)-2)
            
            # Return calculated loss
            return self.criterion(logits_flat, targets_flat)
        elif targets.dim() == 2:
            # Handle 2D targets directly
            return self.criterion(logits.flatten(start_dims=len(logits.shape)-2), targets.flatten())
        else:
            raise ValueError(f"Unexpected targets shape: {targets.shape}")
    
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        """Main model entry point handling both inference and training"""
        # Enforce expected format for hidden_state (unused parameter)
        if hidden_state is not None:
            # Only encoder matters, decoder will reset state anyway
            pass
            
        images = images.to(self.device, dtype=torch.float32)
        
        # Get memory features from encoder
        memory = self.encode_images(images)
        
        # Determine output
        if captions is not None:
            # Training/forced generation: compute losses normally
            inputs = captions[:, :-1] if captions.dim()==2 else captions[:, :, :-1]
            targets = captions[:, 1:] if captions.dim()==2 else captions[:, :, 1:]
            
            # Decode the sequence
            logits, _ = self.decode_sequence(inputs, memory)
            
            # Align shapes correctly for output
            logits = logits.reshape(-1, self.vocab_size)
            
            # Return predicted distribution and intermediate states
            return logits, _

        raise NotImplementedError()