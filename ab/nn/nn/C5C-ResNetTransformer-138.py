import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, q, k, v):
        attn_output, _ = self.attn(q, k, v)
        return attn_output

class CNNEncoder(nn.Module):
    def __init__(self, h_dims=768):
        super().__init__()
        self.h_dims = h_dims
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Conv2d(512, h_dims, kernel_size=1)
        
    def forward(self, images):
        features = self.body(images)
        pooled = self.globalpool(features).flatten(1)
        projected = self.proj(pooled).unsqueeze(1)
        return projected

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.linear3 = nn.Linear(embed_dim, dim_feedforward)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
    def forward(self, tgt, memory, mask=None):
        residual = tgt
        tgt = self.norm1(tgt)
        if mask is not None:
            tgt = self.self_attn(tgt, tgt, tgt, attn_mask=mask)[0]
        tgt = residual + tgt
        residual = tgt
        tgt = self.norm2(tgt)
        tgt = self.cross_attn(tgt, memory, memory)[0]
        tgt = residual + tgt
        tgt = self.linear1(tgt)
        tgt = self.relu(tgt)
        tgt = self.dropout(tgt)
        tgt = self.linear2(tgt)
        return F.relu(tgt)

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, vocab_size):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.pos_decoder = PositionalEncoding(embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads)
            for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, tgt, memory):
        batch_size, seq_len = tgt.size(0), tgt.size(1)
        tgt = self.embedding(tgt) * math.sqrt(self.embed_dim)
        tgt = self.pos_encoder(tgt)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=tgt.device), diagonal=1)
        tgt_mask = torch.where(tgt_mask == 1, -float('Inf'), 0).to(tgt.device)
        
        for layer in self.layers:
            tgt = layer.forward(tgt, memory, mask=tgt_mask)
            
        output = self.fc_out(tgt)
        return output

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.encoder = CNNEncoder(h_dims=640)
        self.decoder = TransformerDecoder(num_layers=6, embed_dim=640, num_heads=8, vocab_size=out_shape)
        self.vocab_size = out_shape
        self.h_dims = 640
        
    def init_zero_hidden(self, batch, device):
        return torch.zeros(batch, self.h_dims).to(device), torch.zeros(batch, self.h_dims).to(device)
        
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm.get('lr', 1e-3))
        
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            if captions.ndim == 3:
                targets = captions[:, :, 0]
            else:
                targets = captions[:, 0]
            inputs = targets[:, :-1]
            targets = targets[:, 1:]
            
            memory = self.encoder(images)
            logits, _ = self.decoder(inputs, memory)
            loss = self.criteria(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
            
    def forward(self, images, captions=None, hidden_state=None):
        memory = self.encoder(images)
        if captions is not None:
            if captions.ndim == 3:
                inputs = captions[:, :, :-1]
                batch_size, seq_len, _ = captions.size()
                targets = captions[:, :, 1:]
            else:
                inputs = captions[:, :-1]
                targets = captions[:, 1:]
                
            embedded = inputs.unsqueeze(1).expand(batch_size, seq_len, self.h_dims)
            embedded = self.embedding_layer(embedded)
            embedded = self.pos_encoder(embedded)
            mask = torch.triu(torch.full((seq_len, seq_len), fill_value=-float('Inf'), device=captions.device), diagonal=1)
            
            hidden_state_in = hidden_state if hidden_state is not None else self.init_zero_hidden(inputs.size(0), self.device)
            hidden = hidden_state_in[0]
            
            for i in range(seq_len):
                embedded_i = embedded[:, i:i+1]
                x = embedded_i + hidden
                
                hidden = self.gru_cell(x, hidden)
                
            logits, hidden = self.linear_fc(F.relu(hidden))
            return logits.permute(1, 0, 2), hidden.permute(0, 1, 2)
            
        else:
            raise NotImplementedError()
            
    def embedding_layer(self, x):
        return self.decoder_embedding(x)
        
    def gru_cell(self, x, hidden):
        x = x + hidden
        gate_weights = self.gru(x)
        reset, update = gate_weights.chunk(2, 1)
        hidden = F.sigmoid(reset) * hidden + F.tanh(update) * F.sigmoid(self.xavier_init(x))
        return hidden

    def linear_fc(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x, self.linear2(x)