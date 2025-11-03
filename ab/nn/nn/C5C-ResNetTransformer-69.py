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
        self.vocab_size = out_shape[0]
        self.hidden_dim = 768  # >640
        
        # CNN Encoder producing [B, S, 768]
        self.encoder = nn.Sequential(
            BasicConv2d(in_shape[1], 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(32, 64, kernel_size=3, padding=1, stride=2),
            BasicConv2d(64, 128, kernel_size=3, padding=1),
            BasicConv2d(128, 256, kernel_size=3, padding=1),
            nn.Unflatten(2, torch.Size([int(in_shape[2]/8), int(in_shape[3]/8)])),
            nn.Flatten(start_dim=2)
        )
        
        # Dynamic decoder selection based on supported hyperparameters
        num_dec_layers = 6
        dec_head_size = 8
        if dec_head_size <= hidden_dim // (num_dec_layers * dec_head_size):
            # Transformer decoder option
            self.decoder = TransformerDecoder(
                self.hidden_dim, 
                num_dec_layers=num_dec_layers,
                num_heads=dec_head_size,
                vocab_size=self.vocab_size
            )
        else:
            # LSTM/GRU fallback option (though better to use Transformer)
            self.decoder = LSTMDecoder(
                self.hidden_dim,
                vocab_size=self.vocab_size
            )
            
    def init_zero_hidden(self, batch: int, device: torch.device):
        return (None, None)
        
    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1).to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=max(float(prm.get('lr', 1e-3)), 3e-4))
        
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            memory = self.encoder(images)
            logits, _ = self.decoder(inputs, memory)
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
        
class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, inputs, memory):
        # inputs: [B, T]
        # memory: [B, S, H]
        
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        emb = self.embed(inputs)  # [B, T, H]
        emb = emb + memory[:,:,:]  # Add memory as additional context
        
        # Initialize hidden state/cell state properly
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(memory.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(memory.device)
        hidden = (h0, c0)
        
        output, hidden = self.lstm(emb, hidden)
        return self.linear(output), hidden[0]

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int = 2048, batch_first: bool = True):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.ff1 = nn.LayerNorm(d_model)
        self.ff2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU(True)
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)

class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int, vocab_size: int):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = None  # Can add later if needed
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, inputs, memory):
        # inputs: [B, T] -> shape [T, B, d_model] internally
        # memory: [B, S, d_model]
        
        seq_len = inputs.size(1)
        batch_size = inputs.size(0)
        x = self.embedding(inputs).view(seq_len, batch_size, self.d_model)
        
        # Apply cross attention over memory
        for layer in self.layers:
            x, _ = layer.cross_attention(x, memory)
            x = self.activation(x)
            x = self.ff1(x)
            x = F.relu(x)
            x = self.ff2(x)
            
        # Back to [B, T, d_model]
        x = x.view(batch_size, seq_len, self.d_model)
        return self.fc_out(x), x[-1:, :, :]

# Main Net implementation
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        
        # Encoder setup
        self.encoder = nn.Identity()
        memory_dim = 768
        
        # Define encoder blocks
        self.stem_conv = BasicConv2d(in_shape[1], 32, kernel_size=3, padding=1)
        self.layer1 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.layer2 = BasicConv2d(64, 128, kernel_size=3, padding=1)
        self.layer3 = BasicConv2d(128, 256, kernel_size=3, padding=1)
        self.layer4 = BasicConv2d(256, 512, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.reducer = nn.Linear(512, memory_dim)
        
    def forward(self, images):
        x = F.relu(self.stem_conv(images))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.adaptive_pool(x).squeeze(-1).squeeze(-1)
        return self.reducer(x)

    def teacher_forcing_decode(self, inputs, memory):
        # inputs: [B, T-1]
        # memory: [B, S, H]
        out, hidden = self.decoder(inputs, memory)
        return out, hidden
        
    def autoregressive_decode(self, start_token, memory, max_length=20):
        # Standard beam search implementation omitted due to brevity
        # Return generation results
        pass

# Placeholder for supporting classes (assumed to exist elsewhere in the codebase)
class PositionalEncoding:
    # Implementation inherited from captioning architecture examples
    pass
    
class CNNEncoder:
    # Implementation inherited from captioning architecture examples
    pass