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
        
        # Encoder: Simple CNN backbone
        self.encoder = nn.Sequential(
            # Base encoder architecture similar to SqueezeNet v1.1
            nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(64, 16, kernel_size=1),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(128, 48, kernel_size=1),
            nn.Conv2d(48, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(192, 64, kernel_size=1),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.Conv2d(64, 512, kernel_size=3, padding=1)
        )
        
        # Feature reduction/pooling to get [B, S, H] with H ≥ 640
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.reduce = nn.Linear(512, 768)
        
        # Decoder: LSTM-based sequence generator
        self.embeddings = nn.Embedding(self.vocab_size, 768)
        hidden_size = 768
        
        # Conditioned LSTM decoder
        self.rnn = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )
        
        # Projection back to vocabulary
        self.fc_out = nn.Linear(hidden_size, self.vocab_size)
        
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.optimizer = None
    
    def init_zero_hidden(self, batch: int, device: torch.device):
        # Initialize hidden states to zeros
        weight = next(self.parameters())
        hidden_size = weight.size(1)
        return torch.randn(2, batch, hidden_size, device=device)
    
    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = self.criterion.to(self.device)
    
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encode(images)
            hidden_state, _ = self.rnn.init_hidden(len(images), self.device)
            
            output, _ = self.forward(images, inputs, hidden_state)
            loss = self.calculate_loss(output, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        # Extract visual features from images
        feat = self.encoder(images)
        pooled_feat = self.pool(feat).squeeze(-1).squeeze(-1)  # [B, 512] -> [B, 1, 512]
        return self.reduce(pooled_feat).unsqueeze(1)  # [B, 1, 768]
    
    def calculate_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate cross entropy loss ignoring padding index (typically 0)
        return self.criterion(pred.reshape(-1, self.vocab_size), target.reshape(-1))
    
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor]=None, hidden_state=None):
        # Only move data to device once
        images = images.to(self.device, dtype=torch.float32)
        
        # Get memory features from images
        memory = self.encode(images)
        
        # Process captions if provided
        if captions is not None:
            caps = captions[:,0,:].long().to(self.device)
            inputs = caps[:, :-1]
            targets = captions[:, 1:] if captions.ndim == 3 else captions
            
            # Embed captions
            emb = self.embeddings(inputs)
            emb_with_pos = self.apply_positional_encoding(emb)
            
            # Run through LSTM decoder
            output, hidden_state = self.rnn(emb_with_pos, hidden_state, memory)
            
            # Project to vocabulary distribution
            logits = self.fc_out(output)
            
            return logits, hidden_state
        
        # Return memory features alone if no captions
        return memory
    
    def apply_positional_encoding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply learned positional encoding to input embeddings."""
        max_len = embeddings.size(1)
        d_model = embeddings.size(-1)
        
        pos_encoding = torch.zeros(max_len, d_model).to(embeddings.device)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return embeddings + pos_encoding.unsqueeze(0)
    
    def decode(self, images: torch.Tensor, max_length: int=20) -> List[str]:
        """Generate captions using greedy decoding."""
        memory = self.encode(images)
        hidden = self.init_zero_hidden(images.size(0), self.device)
        
        # Start with <SOS> token (we'll need proper SOS/EOS handling)
        # SOS token index should be provided externally
        sos_idx = self.vocab_size - 2  # Assuming SOS is the second last token
        tokens = torch.ones(images.size(0), 1).fill_(sos_idx).long().to(self.device)
        
        # Continue generating until we hit max_length or <EOS>
        eos_idx = self.vocab_size - 1  # EOS token index
        
        for _ in range(max_length):
            # Break if we're done or max length exceeded
            if tokens[-1,-1] == eos_idx:
                break
                
            # Get last word and update hidden state
            last_word = tokens[:,-1:]
            last_emb = self.embeddings(last_word)
            last_emb_pe = self.apply_positional_encoding(last_emb)
            
            # Predict next step
            next_output, hidden = self.rnn(last_emb_pe, hidden, memory)
            next_logits = self.fc_out(next_output)
            
            # Take argmax across vocabulary
            next_tokens = last_word.repeat(1, next_logits.size(1)).gather(1, next_logits.argmax(dim=-1))
            
            # Append predicted token
            tokens = torch.cat([tokens, next_tokens], dim=-1)
            if tokens[-1,-1] == eos_idx:
                break
                
        return tokens.tolist()