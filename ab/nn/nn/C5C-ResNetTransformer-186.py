import torch
import torch.nn as nn
import math
from typing import Tuple, Dict, Any

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        pe = torch.zeros(max_len, 1, d_model)
        pe[0, 0] = 1
        for pos in range(1, max_len):
            pe[pos, 0, 0::2] = torch.sin(torch.tensor(pos) * torch.tensor(1000.0**(-float(i)/d_model) for i in range(0, d_model//2, 2)))
            pe[pos, 0, 1::2] = torch.cos(torch.tensor(pos) * torch.tensor(1000.0**(-float(i)/d_model) for i in range(0, d_model//2, 2)))
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Net(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], out_shape: Tuple[int, int, int], prm: Dict[str, Any], device: torch.device):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        
        # Encoder
        self.encoder = self.build_encoder(in_shape, out_shape, prm)
        
        # Decoder
        self.decoder = self.build_decoder(prm)
        
        # supported_hyperparameters
        self.supported_hyperparameters = supported_hyperparameters

    def build_encoder(self, in_shape: Tuple[int, int, int], out_shape: Tuple[int, int, int], prm: Dict[str, Any]) -> nn.Module:
        # Define a simple CNN encoder
        channels, height, width = in_shape
        d_model = out_shape[2]
        
        stem = nn.Sequential(
            nn.Conv2d(channels, 64, 7, 2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        
        # Calculate output dimensions after stem
        output_dim = 512 * (height // 4) * (width // 4)  # Assuming 224x224 input and several downsampling steps
        
        # Project to desired hidden dimension
        self.projector = nn.Linear(output_dim, d_model)
        
        # Flatten the spatial dimensions
        self.flatten = nn.Flatten()
        
        return stem

    def build_decoder(self, prm: Dict[str, Any]) -> nn.Module:
        # Define a simple RNN decoder
        d_model = prm.get('d_model', 768)
        vocab_size = prm.get('vocab_size', 10000)
        hidden_size = d_model
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
        return self.rnn

    def train_setup(self, train_data: torch.Tensor) -> None:
        # Set up the model for training
        pass

    def learn(self, train_data: torch.Tensor) -> torch.Tensor:
        # Teacher forcing
        captions = train_data[:, :, 0]  # Assume captions are provided
        images = train_data[:, :, 1]    # Assume images are provided
        
        # Get memory from encoder
        memory = self.encoder(images)
        
        # Get initial hidden state from memory
        batch_size = images.size(0)
        initial_hidden = memory[:, 0, :].unsqueeze(0)  # [1, B, H]
        
        # Forward through decoder
        logits, hidden_state = self.decoder(captions, initial_hidden)
        
        # Reshape logits to match expected shape
        logits = logits.permute(0, 2, 1)  # [B, T, H]
        logits = self.fc_out(logits)      # [B, T, vocab_size]
        
        # Assert shape
        assert logits.shape == captions.shape[1:], "Logits shape does not match expected shape"
        
        return logits

    def forward(self, inputs: torch.Tensor, hidden_state: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process inputs through the decoder
        if hidden_state is not None:
            # If hidden_state is provided, use it
            output, hidden_state = self.rnn(inputs, hidden_state)
        else:
            # Otherwise, initialize hidden_state
            batch_size = inputs.size(0)
            hidden_state = torch.zeros(1, batch_size, self.rnn.hidden_size).to(inputs.device)
            output, hidden_state = self.rnn(inputs, hidden_state)
        
        # Condition on features (memory)
        # In this implementation, we don't explicitly condition on features during decoding
        # but we can use them to initialize the hidden_state
        
        # Project output to match features dimension
        output = output.permute(0, 2, 1)  # [B, T, H]
        logits = self.fc_out(output)      # [B, T, vocab_size]
        
        # Return logits and updated hidden_state
        return logits, hidden_state

def supported_hyperparameters() -> Dict[str, Any]:
    return {'lr', 'momentum'}

if __name__ == '__main__':
    # Test the code
    in_shape = (3, 224, 224)
    out_shape = (1, 1, 768)
    prm = {'d_model': 768, 'vocab_size': 10000}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = Net(in_shape, out_shape, prm, device)
    print(net)