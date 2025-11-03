import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def supported_hyperparameters():
    return {'lr','momentum'}


class CNNEncoder(nn.Module):
    """CNN Encoder that extracts image features"""
    def __init__(self, in_channels: int, embed_dim: int, num_layers: int):
        super(CNNEncoder, self).__init__()
        # Initial convolution
        self.stem = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=7, stride=2, padding=3
        )
        
        # Encoder layers (using multiple dense blocks)
        self.transition = nn.Sequential()
        self.dense_blocks = nn.ModuleList()
        
        # Define features for each dense block
        for i in range(num_layers):
            reduction_factor = 2 ** i
            self.dense_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        embed_dim * reduction_factor, 
                        embed_dim * reduction_factor * 4,
                        kernel_size=3, padding=1
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(embed_dim * reduction_factor * 4),
                    nn.Dropout(0.5)
                )
            )
            
            if i < num_layers - 1:
                self.transition.append(
                    nn.Sequential(
                        nn.Conv2d(
                            embed_dim * reduction_factor * 4, 
                            embed_dim * reduction_factor,
                            kernel_size=1
                        ),
                        nn.BatchNorm2d(embed_dim * reduction_factor),
                        nn.ReLU(),
                        nn.Dropout(0.5)
                    )
                )

        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x: torch.Tensor):
        # Process through layers
        out = self.stem(x)
        for block in self.dense_blocks:
            out = block(out)
        for transition in self.transition:
            out = transition(out)
        
        # Extract feature vector (without spatial dimensions)
        return out.mean([2, 3])

class Decoder(nn.Module):
    """LSTM Decoder conditioned on encoder features"""
    def __init__(self, embed_dim: int, vocab_size: int):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, features, captions, **kwargs):
        # Shape asserts
        batch_size = captions.size(0)
        seq_length = captions.size(1)
        assert features.shape == (batch_size, self.embed_dim), f"Expected features shape (batch_size, embed_dim) but got {features.shape}"
        assert captions.shape == (batch_size, seq_length), f"Expected captions shape (batch_size, seq_length) but got {captions.shape}"

        # Embed the captions
        embedded = self.embedding(captions)   # (batch_size, seq_length, embed_dim)

        # Set initial hidden state to features (expanded to (1, batch_size, embed_dim))
        h0 = features.unsqueeze(0)   # (1, batch_size, embed_dim)

        # We don't have a cell state, so we set it to zeros.
        c0 = torch.zeros(1, batch_size, self.embed_dim)

        # Run the LSTM
        output, _ = self.lstm(embedded, (h0, c0))   # output: (batch_size, seq_length, embed_dim)

        # Apply linear layer to get the output probabilities for each token at each time step
        output = self.linear(output)   # (batch_size, seq_length, vocab_size)

        return output

class Net(nn.Module):
    """Main network combining encoder and decoder"""
    def __init__(self, in_channels, embed_dim, num_layers, vocab_size):
        super(Net, self).__init__()
        self.encoder = CNNEncoder(in_channels, embed_dim, num_layers)
        self.decoder = Decoder(embed_dim, vocab_size)

    def train_setup(self, device, dtype, **kwargs):
        """Set up the model for training"""
        self.device = device
        self.dtype = dtype
        self.to(device, dtype)

    def learn(self, x, y, **kwargs):
        """Training method (placeholder)"""
        pass

    def forward(self, x, y, **kwargs):
        """Main forward pass with teacher forcing"""
        # x is the input image, y is the captions (with teacher forcing)
        # We first pass x through the encoder
        features = self.encoder(x)   # (batch_size, embed_dim)

        # Then we pass features and y through the decoder
        output = self.decoder(features, y, **kwargs)   # (batch_size, seq_length, vocab_size)

        return output