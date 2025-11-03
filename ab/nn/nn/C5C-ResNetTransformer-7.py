import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        return self.flatten(x)

class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Encoder: a simple CNN
        self.encoder = self.build_encoder(args)
        # Decoder: a GRU
        self.decoder = self.build_decoder(args)
        self.hidden_size = args.hidden_size  # This is the hidden size for the decoder, must be >=640

    def build_encoder(self, args):
        # We'll create a simple CNN encoder.
        # Input: [B, C, H, W]
        # Output: [B, 1, 640] (one token of 640 dimensions)
        in_channels = args.in_channels
        out_features = args.hidden_size  # 640

        layers = []
        layers.append(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))

        layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))

        layers.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU())
        layers.append(nn.AvgPool2d(4))  # Global average pooling to 1x1
        layers.append(FlattenLayer())
        layers.append(nn.Linear(256, out_features))

        return nn.Sequential(*layers)

    def build_decoder(self, args):
        # We'll use a GRU decoder.
        # Input: [B, T_in]
        # Output: [B, T_out, vocab_size]
        embedding_dim = args.hidden_size  # 640
        hidden_dim = args.hidden_size
        num_layers = args.num_layers
        dropout = args.dropout

        # Embedding layer for the vocabulary
        vocab_size = args.vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # GRU layers
        self.gru = nn.GRU(input_size=embedding_dim + hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        # Linear layer to output the vocabulary distribution
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def train_setup(self, args):
        # This function is called during training setup.
        # We might need to initialize the embedding layer or something.
        pass

    def learn(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None) -> Tuple[torch.Tensor, Optional[ Tuple[torch.Tensor, torch.Tensor] ]]:
        # If captions is not None, we are in training mode.
        if captions is not None:
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            memory = self.encoder(images)  # [B, 1, 640]
            # Now, run the decoder on the inputs and memory.
            # The decoder expects inputs [B, T_in] and memory [B, 1, 640] (we'll use memory as part of the context)
            # We'll use the memory to condition the GRU at each time step.

            # Expand the memory to match the sequence length of inputs.
            expanded_memory = memory.expand(-1, inputs.size(1), -1)  # [B, T_in, 640]

            # Concatenate memory and embedded input along the feature dimension.
            inputs_embedded = self.embedding(inputs)  # [B, T_in, 640]
            inputs_embedded = torch.cat((inputs_embedded, expanded_memory), dim=-1)  # [B, T_in, 1280]

            # Now, run the GRU.
            # The GRU expects hidden_state of shape [num_layers, batch_size, hidden_size]
            # If hidden_state is None, we initialize it.
            if hidden_state is None:
                # Initialize hidden_state
                hidden_state = self.init_hidden(images.size(0))

            # The GRU returns output [B, T_in, hidden_size] and hidden_state [num_layers, B, hidden_size]
            output, hidden_state = self.gru(inputs_embedded, hidden_state)
            logits = self.fc_out(output)

            return logits, hidden_state
        else:
            # During inference, we might need to generate captions step by step.
            # But the problem only requires teacher forcing for the learn method.
            # We'll return None for logits and hidden_state.
            return None, None

    def init_hidden(self, batch_size):
        # For GRU, hidden_state is [num_layers, batch_size, hidden_size]
        # We'll set num_layers=1, so [1, batch_size, hidden_size]
        return torch.zeros(1, batch_size, self.hidden_size)

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None) -> Tuple[torch.Tensor, Optional[ Tuple[torch.Tensor, torch.Tensor] ]]:
        # This is the same as learn, but without the conditioning on captions.
        if captions is not None:
            return self.learn(images, captions, hidden_state)
        else:
            # Inference mode: we need to generate captions.
            # We'll use the decoder in an auto-regressive manner.
            # But the problem does not require this, so we'll return None for now.
            # However, the API requires the same signature.
            # Let's return None for logits and hidden_state.
            return None, None

    def init_zero_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # This function initializes the hidden state for the decoder.
        # For GRU, it is a single tensor of shape [1, batch_size, hidden_size]
        return self.init_hidden(batch_size)

def supported_hyperparameters():
    return {'lr','momentum'}