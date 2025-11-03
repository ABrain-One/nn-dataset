import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.encoder = None
        self.rnn = None
        self.embedding = None

    def train_setup(self):
        pass

    def learn(self, images, captions):
        pass

    def forward(self, images, captions=None, hidden_state=None):
        pass

    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.empty(0, device=device), torch.empty(0, device=device)

    def supported_hyperparameters():
    return {'lr','momentum'}



# --- auto-closed by AlterCaptionNN ---