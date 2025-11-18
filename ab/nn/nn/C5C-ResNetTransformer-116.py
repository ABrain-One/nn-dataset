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
        self.vocab_size = out_shape
        
        # Encoder Architecture (CNN based)
        enc_in_c = int(in_shape[1])
        self.encoder = nn.Sequential(
            nn.Conv2d(enc_in_c, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 768)
        )
        
        # Decoder Architecture (Transformer)
        dec_in_dim = 768  # Must match encoder output dimension
        self.hidden_dim = 768  # >=640

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(encoder_layers, num_layers=6)
        
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)
        
        self.optimizer = None

    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.empty(0, device=device), torch.empty(0, device=device)

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm.get('lr', 1e-3))

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            memory = self.encoder(images)
            logits, _ = self.rnn_forward(inputs, None, memory)
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def rnn_forward(self, inputs, hidden_state=None, memory=None):
        # Replicate memory across sequence length (for Transformer cross-attention)
        expanded_memory = memory.transpose(0, 1)  # Shape [S, B, H] → [B, S, H] remains same? Actually [B, 1, H] initially
        
        # Prepare input embeddings
        embedded = self.embedding(inputs)
        embedded = embedded.transpose(0, 1)  # [B, T, H]
        
        # Create transformer decoder object
        dec = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=False,
            dropout=0.1
        )
        
        # Forward pass through transformer decoder
        out = dec(embedded, expanded_memory, memory_key_padding_mask=None)
        
        # Final prediction head
        logits = self.fc_out(out)
        return logits, None

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        
        if captions is not None:
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            out = self.rnn_forward(inputs, None, memory)
            return out, None
            
        else:
            raise NotImplementedError()

    def decode(self, images, captions=None, hidden_state=None, max_length=20):
        """Generate caption using beam search"""
        memory = self.encoder(images)
        hidden_state = None
        
        # Start with SOS token
        input_sequence = torch.full((images.size(0), 1), self.vocab_size-1, device=self.device, dtype=torch.long)
        
        for t in range(max_length):
            logits, _ = self.rnn_forward(input_sequence, hidden_state, memory)
            next_tokens = logits.argmax(dim=-1)
            if next_tokens.item() == 0:  # EOS token found
                break
            input_sequence = torch.cat([input_sequence, next_tokens.unsqueeze(1)], dim=1)
        
        return input_sequence