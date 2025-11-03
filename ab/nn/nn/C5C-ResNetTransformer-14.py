import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape)
        
        # Encoder: CNN with projected features
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(128 * (in_shape[2]//4) * (in_shape[3]//4), 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
        
        # Decoder: LSTM with attention
        self.embedding = nn.Embedding(self.vocab_size, 768)
        self.lstm = nn.LSTM(input_size=768 + 768, hidden_size=768, batch_first=True, num_layers=2)
        self.fc_out = nn.Linear(768, self.vocab_size)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.optimizer = None

    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.randn(2, batch, 768).to(device), torch.randn(2, batch, 768).to(device)

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criterion = self.criterion.to(self.device)
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encoder(images)
            embedding = self.embedding(inputs)
            stacked = torch.cat([embedding, memory.repeat((images.size(0), inputs.size(1), 1))], dim=-1)
            
            output, _ = self.lstm(stacked, None)
            logits = self.fc_out(output)
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        
        if captions is not None:
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            embedding = self.embedding(inputs)
            stacked = torch.cat([embedding, memory.repeat((memory.size(0), inputs.size(1), 1))], dim=-1)
            output, hidden_state = self.lstm(stacked, hidden_state)
            logits = self.fc_out(output)
            
            assert logits.shape == inputs.shape[1]
            assert logits.shape[-1] == self.vocab_size
            
            return logits, hidden_state
        else:
            raise NotImplementedError()