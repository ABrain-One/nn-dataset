import torch
import torch.nn as nn
from typing import Tuple, Optional, List

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.hidden_dim = 768
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, self.hidden_dim),
            nn.Unsqueeze(1)
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=4),
            num_layers=6,
            batch_first=True
        )
        self.embedding = nn.Embedding(out_shape[0], self.hidden_dim)
        self.projection = nn.Linear(self.hidden_dim, out_shape[0])

    def train_setup(self, optimizer, criterion, hparams):
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device if 'device' in hparams else torch.device('cuda' if hparams.get('use_gpu', True) else 'cpu')
        self.to(self.device)

    def learn(self, images, captions):
        memory = self.encoder(images)
        if captions.ndim == 3:
            tgt_input = captions[:, :-1]
        else:
            tgt_input = captions[:, :-1]
        embedded = self.embedding(tgt_input)
        output = self.decoder(embedded, memory)
        logits = self.projection(output)
        return logits, None

    def forward(self, images, captions=None, hidden_state=None):
        if captions is not None:
            if captions.ndim == 3:
                tgt_input = captions[:, :-1]
            else:
                tgt_input = captions[:, :-1]
            memory = self.encoder(images)
            embedded = self.embedding(tgt_input)
            output = self.decoder(embedded, memory)
            logits = self.projection(output)
            batch_size = images.size(0)
            if hidden_state is None:
                hidden_state = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
            else:
                hidden_state = hidden_state
            return logits, hidden_state

def supported_hyperparameters():
    return {'lr','momentum'}


# --- auto-closed by AlterCaptionNN ---