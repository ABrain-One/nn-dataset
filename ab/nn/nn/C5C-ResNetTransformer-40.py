import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class EncoderCNN(nn.Module):
    def __init__(self, in_channels=3, hidden_size=768):
        super(EncoderCNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        self.layer1 = self._conv_block(in_channels, 64)
        self.layer2 = self._conv_block(64, 128)
        self.layer3 = self._conv_block(128, 256)
        self.layer4 = self._conv_block(256, 512)
        self.layer5 = self._conv_block(512, 1024)
        self.layer6 = self._conv_block(1024, hidden_size)
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

class DecoderLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, vocab_size):
        super(DecoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, hidden_state, memory):
        if hidden_state is None:
            hidden_state = torch.zeros(self.num_layers, inputs.size(0), self.hidden_dim, device=inputs.device)
        
        embedded = self.embedding(inputs)
        memory = memory.repeat(embedded.size(1), 1, 1)
        
        cross_attn_output, _ = self.cross_attention(embedded, memory, memory)
        embedded = embedded + cross_attn_output
        
        outputs, hidden_state = self.lstm(embedded, hidden_state)
        logits = self.fc_out(outputs)
        
        return logits, hidden_state

class Net(nn.Module):
    def __init__(self, encoder: EncoderCNN, decoder: DecoderLSTM, vocab_size: int, device: torch.device):
        super(Net, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.device = device

    def train_setup(self, lr: float, momentum: float):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def learn(self, train_data: Tuple[torch.Tensor, torch.Tensor]):
        images, captions = train_data
        if captions.ndim == 3:
            captions = captions[:, :-1]
        else:
            raise ValueError("Captions must be 3D")
        
        memory = self.encoder(images)
        logits, hidden_state = self.decoder(captions, None, memory)
        
        # Calculate loss (not implemented, but would be done)
        loss = torch.tensor([0.0], device=logits.device)
        return loss

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state: Optional[torch.Tensor] = None):
        if captions is not None:
            if captions.ndim == 3:
                captions = captions[:, :-1]
            elif captions.ndim == 2:
                captions = captions[:, :-1]
            else:
                raise ValueError("Captions must be 2D or 3D")
            
            memory = self.encoder(images)
            logits, hidden_state = self.decoder(captions, hidden_state, memory)
            return logits, hidden_state
        else:
            # Beam search implementation
            raise NotImplementedError()

def supported_hyperparameters():
    return {'lr','momentum'}