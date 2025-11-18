import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def supported_hyperparameters():
    return {'lr','momentum'}


class CNNEncoder(nn.Module):
    def __init__(self, input_channels=3, output_dim=768):
        super().__init__()
        self.initial_block = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pooling_block = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, output_dim)

    def forward(self, images):
        x = self.initial_block(images)
        x = self.pooling_block(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, hidden_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def init_zero_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.gru.hidden_size).to(self.embedding.weight.device)
        
    def forward(self, inputs, hidden_state, memory):
        # Expand memory to match sequence length
        expanded_memory = memory.unsqueeze(0).expand(-1, inputs.size(1), -1)
        # Repeat memory across batch if necessary, ensuring broadcasting works
        memory_broadcast = expanded_memory
        
        # Combined input: concat embedded word and memory at each time step
        embedded = self.embedding(inputs)
        combined = torch.cat([embedded, memory_broadcast], dim=-1)
        
        output, hidden_state = self.gru(combined, hidden_state)
        logits = self.fc_out(output)
        return logits, hidden_state

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape[0])
        self.hidden_dim = 640  # Minimum dimension requirement
        self.embed_dim = 768
        self.input_channels = in_shape[1]
        self.output_channels = out_shape[0]
        
        # Create encoder
        self.encoder = CNNEncoder(
            input_channels=in_shape[1],
            output_dim=self.embed_dim
        )
        
        # Create decoder (choose GRU or LSTM)
        self.rnn_type = 'gru'
        if 'rnn_type' in prm and prm['rnn_type'] in ['lstm', 'gru']:
            self.rnn_type = prm['rnn_type'].lower()
        else:
            # Default to gru
            self.rnn_type = 'gru'
            
        if self.rnn_type == 'gru':
            self.decoder = DecoderGRU(
                vocab_size=out_shape[0],
                embedding_dim=self.embed_dim,
                hidden_dim=self.embed_dim
            )
        else:  # lstm
            self.decoder = nn.LSTM(
                input_size=self.embed_dim + self.embed_dim,
                hidden_size=self.embed_dim,
                num_layers=1,
                batch_first=True
            )
            # Define proper initialization method for LSTM
            def init_lstm_hidden(batch_size):
                # For LSTM we need (h, c), both hidden_states
                h = torch.zeros(1, batch_size, self.embed_dim).to(device)
                c = torch.zeros(1, batch_size, self.embed_dim).to(device)
                return h, c
                
            self.init_hidden = lambda batch: init_lstm_hidden(batch)
        # Set final embedding and hidden dimension to output_dim
        if self.rnn_type == 'gru':
            self.init_hidden = lambda batch: torch.zeros(1, batch, self.embed_dim).to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
    def init_zero_hidden(self, batch: int, device: torch.device):
        # Depending on decoder type
        if hasattr(self, 'decoder') and hasattr(self.decoder, 'init_zero_hidden'):
            return self.decoder.init_zero_hidden(batch)
        else:
            return torch.zeros(1, batch, self.embed_dim).to(device)
            
    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criteria = self.criterion.to(self.device)
        lr = max(float(prm.get('lr', 1e-3)), 3e-4) if 'lr' in prm else 1e-3
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99) if 'momentum' in prm else 0.9
        
        # Check if adamw optimizer is requested or use default
        if 'optimizer' in prm and prm['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))
    
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encoder(images)
            if self.rnn_type == 'gru':
                logits, _ = self.decoder(inputs, None, memory)
            else:  # lstm
                logits, _, _ = self.decoder(inputs, None, memory)
                
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
            
            if hidden_state is None and self.rnn_type == 'gru':
                hidden_state = self.init_zero_hidden(images.size(0), self.device)
            elif hidden_state is None and self.rnn_type == 'lstm':
                # Return properly formatted hidden state
                batch_size = images.size(0)
                hidden_state_h, hidden_state_c = self.init_hidden(batch_size)
                hidden_state = (hidden_state_h, hidden_state_c)
                
            if self.rnn_type == 'gru':
                logits, hidden_state = self.decoder(inputs, hidden_state, memory)
            else:  # lstm
                logits, hidden_state_h, hidden_state_c = self.decoder(inputs, hidden_state, memory)
                # Return only hidden state as tuple (h,c) for consistency
                hidden_state = (hidden_state_h, hidden_state_c)
                
            assert logits.shape == inputs.shape
            assert logits.shape[-1] == self.vocab_size
            
            # Return both the output and the hidden state
            return logits, hidden_state
        else:
            raise NotImplementedError("Generation mode not implemented")