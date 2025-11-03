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
        in_channels = int(in_shape[1])
        self.hidden_dim = 640
        
        # TODO: Replace self.encoder with custom encoder producing memory tensor [B, S, H] where H >= 640
        self.encoder = nn.Identity()
        
        # TODO: Replace self.rnn with custom decoder implementing forward(inputs, hidden_state, features) -> (logits, hidden_state)
        self.rnn = nn.Identity()
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.empty(0, device=device), torch.empty(0, device=device)

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criteria = (self.criterion,)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm.get('lr', 1e-3))

    def learn(self, train_data):
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encoder(images)
            outputs, _ = self.rnn(inputs, None, memory)
            loss_val = self.criteria[0](outputs.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad()
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        
        if captions is not None:
            cap_inputs = captions[:, :-1]
            logits, hidden_state = self.rnn(cap_inputs, hidden_state, memory)
            logits = logits.permute(1, 0, 2)  # Shape [T-1, B, vocab_size] to [B, T-1, vocab_size]
            return logits, hidden_state
        else:
            raise NotImplementedError("Generation functionality needs to be implemented")