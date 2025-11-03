import torch
import torch.nn as nn
from typing import Optional, Tuple

class View(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.view_args = args
        
    def forward(self, x):
        return x.view(*self.view_args)

class Net(nn.Module):
    def __init__(self, in_shape: Tuple, out_shape: Tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        
        # Encoder: ResNet-like architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 768, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(768),
            nn.AdaptiveAvgPool2d((1, 1)),
            View(-1, 768)
        )
        
        # Decoder: LSTM with embedding
        self.embedding = nn.Embedding(out_shape[0], 768)
        self.rnn = nn.LSTM(input_size=768, hidden_size=768, num_layers=1, batch_first=True)
        self.fc_out = nn.Linear(768, out_shape[0])
        
        # Initialize hidden_state to zeros
        self.hidden_state = None

    def init_zero_hidden(self, batch_size: int):
        # Initialize hidden_state to zeros
        self.hidden_state = None

    def train_setup(self, **kwargs):
        # Set up training parameters
        self.train()

    def learn(self, **kwargs):
        # Continue training
        self.train()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        # First, get the memory from the encoder
        memory = self.encoder(images)
        
        # If captions is provided, then we are in teacher forcing mode.
        if captions is not None:
            # Convert captions to long and to device
            if captions.ndim == 3:
                caps = captions[:,0,:].long().to(self.device)
            else:
                caps = captions.long().to(self.device)
                
            # Get the inputs (all but the last token)
            inputs = caps[:, :-1]
            
            # Embed the inputs
            embedded = self.embedding(inputs)
            
            # If hidden_state is None, initialize it to zeros
            if hidden_state is None:
                hidden_state = None
                
            # Run the LSTM
            outputs, hidden_state = self.rnn(embedded, hidden_state, memory)
            
            # Compute the logits
            logits = self.fc_out(outputs)
            
            # Return the logits and the updated hidden_state
            return logits, hidden_state
        
        # If captions is None, then we are in inference mode.
        # We'll return the memory and the initial hidden_state (if any)
        return memory, self.hidden_state if hidden_state is not None else None

def supported_hyperparameters():
    return {'lr', 'momentum'}