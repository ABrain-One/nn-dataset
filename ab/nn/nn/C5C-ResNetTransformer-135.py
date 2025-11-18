import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=768):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, hidden_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x).squeeze(3).squeeze(2)
        x = self.fc(x)
        return x.unsqueeze(1)

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, memory, hidden_state=None):
        embedded = self.embedding(tgt)
        embedded = embedded.permute(1, 0, 2)
        
        if hidden_state is None:
            hidden_state = (torch.zeros(1, embedded.size(1), self.hidden_dim),
                           torch.zeros(1, embedded.size(1), self.hidden_dim))
        
        output, hidden_state = self.lstm(embedded, hidden_state)
        logits = self.fc_out(output)
        return logits, hidden_state

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.vocab_size = out_shape[0]
        self.encoder = CNNEncoder(hidden_dim=768)
        self.decoder = LSTMDecoder(vocab_size=self.vocab_size, hidden_dim=768)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)

    def learn(self, train_data):
        images, captions = train_data
        images = images.to(self.device, dtype=torch.float32)
        captions = captions.to(self.device)
        
        memory = self.encoder(images)
        inputs = captions[:, :-1]
        targets = captions[:, 1:]
        
        logits, _ = self.decoder(inputs, memory, None)
        loss = self.criterion(logits.view(-1, self.vocab_size), targets.view(-1))
        return loss

    def forward(self, images, captions=None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        
        if captions is not None:
            if captions.ndim == 3:
                captions = captions[:, 0, :]
            inputs = captions[:, :-1]
            logits, hidden_state_new = self.decoder(inputs, memory, hidden_state)
            return logits, hidden_state_new
        else:
            # Beam search implementation would go here
            raise NotImplementedError()

def supported_hyperparameters():
    return {'lr', 'momentum'}