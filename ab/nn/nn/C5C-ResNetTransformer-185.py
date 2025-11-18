import math
import torch
import torch.nn as nn
import torch.nn.functional as F
def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super(Net, self).__init__()
        self.device = device
        
        # Adjust image size parameterization to support dynamic resolutions
        channels, height, width = 3, 224, 224
        
        # Encoder backbone with adjustable hidden dimension
        self.hidden_dim = max(int(prm.get('hidden_dim', 768)), 640)
        
        # Use global average pooling for simpler implementation
        encoder_layers = [
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        ]
        
        # Adjust feature dimensions to match the requested hidden dimension
        intermediate_size = self.hidden_dim // 2
        encoder_layers.extend([
            nn.Flatten(start_dim=-2),
            nn.Linear(256, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, self.hidden_dim)
        ])
        
        # Final fully-connected layer to prepare outputs for the Transformer decoder
        self.initial_projection = nn.Sequential(*encoder_layers)
        
        # Memory sequence length calculation
        img_size = 224  # Assuming standard 224x224 resolution
        self.mem_seq_len = (img_size // 16)**2  # Approximate number of patches
        
        # Decoder using nn.TransformerDecoder
        decoder_layers = []
        for _ in range(6):
            layer = nn.TransformerDecoderLayer(
                d_model=self.hidden_dim, 
                nhead=min(8, self.hidden_dim // 96),  # Need divisible pairs for MultiheadAttention
                dim_feedforward=max(self.hidden_dim*4, 4096),
                batch_first=True,
                dropout=min(prm.get('decoder_dropout', 0.1), 0.2)
            )
            decoder_layers.append(layer)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=min(8, self.hidden_dim//96)),
            num_layers=len(decoder_layers)
        )
        
        # Projection for final output probabilities
        self.classifier = nn.Linear(self.hidden_dim, out_shape[0])

    def init_zero_hidden(self, batch: int, device: torch.device):
        return (
            torch.zeros(batch, self.mem_seq_len, self.hidden_dim, device=device),
            torch.zeros(batch, self.mem_seq_len, self.hidden_dim, device=device)
        )

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=prm.get('patience', 3), factor=0.5
        )

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:, :, 0].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encode(images)
            decoder_output = self.decode(inputs, memory)
            loss = self.criterion(decoder_output.reshape(-1, decoder_output.size(-1)), targets.reshape(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def encode(self, images):
        """Produces memory features [B, S, H]"""
        # Get feature map and apply global averaging
        base = self.initial_projection(images)
        
        # Expand dimensions to represent patch tokens
        batch_size, channels, height, width = base.shape
        seq_length = height * width
        
        # Reshape to [B, S, H]
        base = base.permute(0, 2, 3, 1).reshape(batch_size, seq_length, channels)
        return base

    def decode(self, inputs, memory):
        """Performs decoding via TransformerDecoder"""
        # Input embedding
        embedding_size = self.hidden_dim
        seq_len = inputs.size(1)
        embedding = inputs @ self.embedding_matrix
        
        # Positional encoding
        pos_encoding = self.position_encoding(embedding)
        
        # Initial memory key/value
        memory_key = memory
        memory_value = memory
        
        # Decode with attention
        decoder_output = self.transformer_decoder(pos_encoding, memory_key, memory_value)
        
        # Final prediction
        logits = self.classifier(decoder_output[:, -1])
        return logits
    
    def forward(self, images, captions=None, hidden_state=None):
        """Main forward pass"""
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encode(images)
        
        if captions is not None:
            batch_size = images.shape[0]
            caps = captions[:,0,:]  # Assume COCO-style padded tensor
                
            # Initialize attention mask if needed
            self.setup_masks(caps.size(1))
                
            # Decoder initialization
            seq_length = caps[:, :-1].shape[1]
            embedding = torch.zeros(batch_size, seq_length, self.hidden_dim).to(self.device)
            
            # Create word embedding matrix (learnable parameters)
            self.embedding_matrix = self.create_embedding_matrix()
            
            # Perform decoding
            decoded = self.decode(embedding, memory)
            return decoded, memory
            
        else:
            raise NotImplementedError()

    def create_embedding_matrix(self):
        return Parameter(torch.rand(self.vocab_size, self.hidden_dim))

    def position_encoding(self, embedded_sequence):
        batch_size, seq_len, dim = embedded_sequence.size()
        
        # Generate positional encoding if not cached yet
        if hasattr(self, 'pos_embeddings') and self.pos_embeddings.shape == embedded_sequence.shape:
            return self.pos_embeddings
            
        pos_encoding = torch.zeros_like(embedded_sequence)
        for j in range(seq_len):
            position = torch.ones(1, 1, 1)
            position[0,0,0] = j
            pos_encoding[:,j,:] = torch.sin(position * self.encoding_factor)
            
        return embedded_sequence + pos_encoding

    @property
    def vocab_size(self):
        return self.classifier.out_features

    def setup_masks(self, seq_len):
        # Create causal mask
        self.causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=self.device),
            diagonal=1
        ).bool()

# Usage example
# model = Net(in_shape=(3, 224, 224), out_shape=(1000,), prm={'hidden_dim': 768}, device=torch.device('cuda'))
# This implements a basic pipeline connecting the encoder and transformer decoder
# Further adjustments may be needed for production use