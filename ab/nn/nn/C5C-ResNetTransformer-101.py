import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def supported_hyperparameters():
    return {'lr','momentum'}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 768)

    def forward(self, images):
        features = self.cnn(images)  # [B, 1024, 7, 7] after all convolutions
        pooled = self.pool(features).flatten(1)  # [B, 1024]
        return self.fc(pooled).unsqueeze(1)  # [B, 1, 768]


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=6, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=2048, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        # tgt: [batch, seq] -> embed: [batch, seq, d_model]
        embedded = self.embedding(tgt)
        embedded = self.pos_encoding(embedded)
        seq_len = tgt.size(1)
        tgt_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(tgt.device)
        # memory: [batch, mem_seq, d_model] (mem_seq=1 from encoder)
        out = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask)
        return self.fc_out(out)  # [batch, seq, vocab_size]


class CustomSEBlock(nn.Module):
    """Custom Squeeze-and-Excitation block."""
    def __init__(self, channel, reduction_ratio=0.5):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, int(channel * reduction_ratio)),
            nn.ReLU(),
            nn.Linear(int(channel * reduction_ratio), channel),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """Forward function."""
        b, c, h, w = x.shape
        y = self.se(x)
        return x * y[:, :, None, None]


class Net(nn.Module):
    """Main Image Captioning Network"""
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape[0])
        self.hidden_dim = 768
        
        # TODO: Replace encoder with modified CNN backbone
        self.encoder = CNNEncoder()
        
        # TODO: Replace decoder with TransformerDecoder
        self.decoder = TransformerDecoder(vocab_size=self.vocab_size, d_model=self.hidden_dim, 
                                          num_layers=6, nhead=8)
        
        # Add SE Block after the final encoder layer
        self.se_block = CustomSEBlock(channel=1024)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.optimizer = None

    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.empty(0, device=device), torch.empty(0, device=device)

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criterion = self.criterion.to(self.device)
        # Use AdamW optimizer as specified in classification examples
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))
        # Include RMSNorm/Stochastic Depth as seen in other architectures if applicable
        # But keeping it simple as per classification examples

    def learn(self, train_data):
        """Train using teacher forcing"""
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:, 0, :].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encoder(images)  # [B, 1, 768]
            logits, _ = self.decoder(inputs, memory)  # [B, T-1, V]
            # Flatten the dimensions for loss calculation
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images, captions=None, hidden_state=None):
        """Implement forward pass according to API requirements"""
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)  # [B, 1, 768]
        
        if captions is not None:
            caps = captions[:, 0, :].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            outputs = self.decode_sequence(inputs, memory)
            # Compute shape for hidden_state (current decoder state)
            hidden_state = self.compute_new_hidden(outputs, memory)
            
            # Assertions from API
            assert outputs.shape == (images.size(0), inputs.shape[1], self.vocab_size)
            assert hidden_state.shape == (images.size(0), 1, self.hidden_dim)
            return outputs, hidden_state
        
        else:
            raise NotImplementedError()

    def decode_sequence(self, inputs, memory):
        """Decode a sequence using teacher forcing"""
        embedded = self.embedding(inputs)  # Already handled in TransformerDecoder
        # Using positional encoding and transformer decoding within the TransformerDecoder
        # Return logits and update hidden state as requested by API
        seq_length = inputs.size(1)
        tgt_mask = torch.triu(torch.full((seq_length, seq_length), float('-inf')), diagonal=1)
        out = self.transformer_decoder(inputs, memory, tgt_mask=tgt_mask)
        # Project to vocabulary space
        logits = self.fc_out(out)
        return logits

    def compute_new_hidden(self, outputs, memory):
        """Compute new hidden state according to API"""
        # Transformer output is shaped [batch, seq, hidden_dim]
        # Memory is still [batch, 1, hidden_dim]
        # Create a new hidden state by concatenating and applying a linear transformation
        combined = torch.cat([outputs[:, -1], memory[:, 0]], dim=-1)
        new_hidden = self.fc(combined)  # Simple projection layer
        new_hidden = new_hidden.view(new_hidden.size(0), -1, self.hidden_dim)
        # According to API, hidden_state should be [batch, 1, hidden_dim] or equivalent
        return new_hidden

