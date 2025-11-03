import torch
import torch.nn as nn
from typing import Optional, Tuple

class Net(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, vocab_size: int, device: str):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = nn.Embedding(vocab_size, 768)
        self.device = device

    def train_setup(self, optimizer, lr, momentum):
        pass

    def learn(self, images: torch.Tensor, captions: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Teacher forcing: pass the entire sequence (except last token) to the decoder.
        # The captions are of shape [B, T]
        embedded = self.embedding(captions)
        # The decoder expects embedded captions and memory
        memory = self.encoder(images)
        output = self.decoder(embedded, memory)
        logits = self.projection(output)
        return logits, None

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # This function is not defined in the original code, but the problem says to keep the same structure.
        # We'll assume it's for setting up the model for training, but the original code didn't have it.
        # Since the problem says to fix only syntax/API issues, we'll leave it as is if it's not defined.
        pass

    def supported_hyperparameters():
    return {'lr','momentum'}


# --- auto-closed by AlterCaptionNN ---