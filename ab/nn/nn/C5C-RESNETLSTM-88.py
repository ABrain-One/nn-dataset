import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any

class Net(nn.Module):
    def __init__(self, in_shape: Tuple, out_shape: Tuple, prm: Dict[str, Any], device: torch.device) -> None:
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3
        self.vocab_size = out_shape[0][0][0] if isinstance(out_shape, (tuple, list)) else int(out_shape[0])
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size
        
        # Backward-compat local aliases (old LLM patterns)
        self.hidden_size = 768
        self.num_layers = 1
        self.dropout = 0.3
        
        # Encoder
        self.cnn_encoder = self._build_encoder()
        
        # Decoder
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout)
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(self.hidden_size, 8, batch_first=True)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_encoder(self):
        # Vision Transformer style encoder
        encoder = nn.Sequential(
            # Stem
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Transformer blocks
            self._make_block(64, 128, 2),
            self._make_block(128, 256, 2),
            self._make_block(256, 512, 2),
            self._make_block(512, 768, 2)
        )
        return encoder
    
    def _make_block(self, in_channels: int, out_channels: int, blocks: int) -> nn.Sequential:
        block = nn.Sequential()
        for i in range(blocks):
            block.add_module(f'block_{blocks}_{i}', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            block.add_module(f'batchnorm_{blocks}_{i}', nn.BatchNorm2d(out_channels))
            block.add_module(f'relu_{blocks}_{i}', nn.ReLU())
            in_channels = out_channels
        return block
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def train_setup(self, prm: Dict[str, Any]) -> None:
        # Set up training parameters
        self.lr = prm.get('lr', 1e-4)
        self.momentum = prm.get('momentum', 0.9)
        self.batch_size = prm.get('batch_size', 32)
        self.num_workers = prm.get('num_workers', 4)
    
    def learn(self, train_data: Any) -> None:
        # Training loop
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):  # Fixed number of epochs for demonstration
            running_loss = 0.0
            for i, (images, captions) in enumerate(train_data):
                images = images.to(self.device)
                captions = captions.to(self.device)
                
                # Forward pass
                features = self.cnn_encoder(images)
                embedded = self.embedding(captions)
                gru_output, _ = self.gru(embedded)
                outputs = self.fc_out(gru_output)
                
                # Calculate loss
                loss = criterion(outputs, captions)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f'Epoch [{epoch+1}/10], Step [{i+1}], Loss: {running_loss/100:.4f}')
                    running_loss = 0.0
    
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        # Extract image features
        image_features = self.cnn_encoder(images)
        
        # If captions are provided, use teacher forcing
        if captions is not None:
            # Embed captions
            embedded = self.embedding(captions)
            
            # If hidden_state is provided, use it, otherwise initialize
            if hidden_state is None:
                hidden_state = torch.zeros(self.num_layers, captions.size(0), self.hidden_size, device=self.device)
            
            # Run GRU
            gru_output, hidden_state = self.gru(embedded, hidden_state)
            
            # Apply attention
            gru_output = gru_output.transpose(0, 1)  # [seq_len, batch, hidden_size]
            gru_output = gru_output.contiguous().view(-1, gru_output.size(-1))  # [seq_len * batch, hidden_size]
            gru_output, _ = self.attention(gru_output, image_features, image_features)  # [seq_len * batch, hidden_size]
            gru_output = gru_output.reshape(-1, captions.size(1), gru_output.size(-1))  # [seq_len, batch, hidden_size]
            gru_output = gru_output.transpose(0, 1)  # [batch, seq_len, hidden_size]
            
            # Final output
            outputs = self.fc_out(gru_output)
            outputs = outputs.transpose(0, 1)  # [batch, seq_len, vocab_size]
            
            # Return outputs and hidden_state
            return (outputs, hidden_state)
        
        # During inference, generate captions
        else:
            # Initialize hidden_state
            if hidden_state is None:
                hidden_state = torch.zeros(self.num_layers, images.size(0), self.hidden_size, device=self.device)
            
            # SOS token
            captions = torch.full((images.size(0), 1), self.vocab_size-1, device=self.device, dtype=torch.long)
            
            # Generate sequence
            max_length = 50  # Fixed maximum length for demonstration
            for i in range(max_length):
                # Embed current captions
                embedded = self.embedding(captions)
                
                # Run GRU
                gru_output, hidden_state = self.gru(embedded, hidden_state)
                
                # Apply attention
                gru_output = gru_output.transpose(0, 1)  # [1, batch, hidden_size]
                gru_output = gru_output.contiguous().view(-1, gru_output.size(-1))  # [batch * seq_len, hidden_size]
                gru_output, _ = self.attention(gru_output, image_features, image_features)  # [batch * seq_len, hidden_size]
                gru_output = gru_output.reshape(gru_output.size(0)//1, -1)  # [batch, hidden_size]
                gru_output = gru_output.unsqueeze(0)  # [1, batch, hidden_size]
                gru_output = gru_output.transpose(0, 1)  # [batch, 1, hidden_size]
                
                # Final output
                outputs = self.fc_out(gru_output)
                
                # Select next token
                next_token = outputs.argmax(dim=-1)[:, -1]
                next_token = next_token.reshape(-1, 1)
                
                # Append to captions
                captions = torch.cat((captions, next_token), dim=1)
                
                # Stop if EOS is reached
                if next_token.item() == self.vocab_size-2:  # EOS token
                    break
            
            # Return generated captions and hidden_state
            return (captions, hidden_state)
    
    def supported_hyperparameters():
    return {'lr','momentum'}



# Example usage
if __name__ == '__main__':
    # Create a dummy model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(in_shape=(3, 224, 224), out_shape=(1000,), prm={'lr': 1e-4, 'momentum': 0.9}, device=device)
    
    # Print model structure
    print(model)
    
    # Test forward pass
    images = torch.randn(1, 3, 224, 224).to(device)
    captions = torch.randint(0, 1000, (1, 50)).to(device)
    outputs, hidden_state = model(images, captions)
    print(f"Outputs shape: {outputs.shape}")