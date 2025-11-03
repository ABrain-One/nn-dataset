import torch.nn as nn
import torch.optim as optim
import torch
from typing import Optional, Tuple

class Net(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], out_shape: int, prm: dict, device: torch.device):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.word2idx = None
        self.idx2word = None
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(192, 768)
        )
        
        # Decoder
        self.decoder = nn.TransformerDecoderLayer(d_model=768, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder, num_layers=6)
        self.fc_out = nn.Linear(768, out_shape)
        
        # Hyperparameters
        self.lr = prm.get('lr', 0.001)
        self.momentum = prm.get('momentum', 0.9)
        self.batch_size = prm.get('batch_size', 1)
        self.seq_length = prm.get('seq_length', 20)
        self.vocab_size = out_shape
        
    def learn(self, train_data):
        # Implement learning logic here
        pass
        
    def init_zero_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # For transformer decoder, we don't need hidden_state
        # But we must return two tensors to match the API
        return (torch.zeros(1, batch_size, self.vocab_size).to(self.device),
                torch.zeros(1, batch_size, self.vocab_size).to(self.device))
                
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.size(0)
        if captions is not None:
            # Teacher forcing
            memory = self.encoder(images)  # [B, 1, 768]
            tgt_input = captions[:, :-1]  # [B, T-1]
            out = self.transformer_decoder(tgt_input, memory)  # [B, T-1, 768]
            logits = self.fc_out(out)  # [B, T-1, vocab_size]
            return logits, hidden_state
        else:
            # Beam search (not implemented)
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999))

            raise NotImplementedError

def supported_hyperparameters():
    return {'lr','momentum'}