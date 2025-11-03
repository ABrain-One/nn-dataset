import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :]
        x = x + pe
        return self.dropout(x)

class ViTEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=16, emb_dim=768, num_layers=6, num_heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        
        # Calculate number of patches
        n_patches = (224 // patch_size) ** 2
        
        # Patch embedding layer
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x shape: [B, C, H, W]
        x = self.proj(x)  # [B, emb_dim, 14, 14]
        x = x.flatten(2)  # [B, emb_dim, 196]
        x = x.transpose(1, 2)  # [B, 196, emb_dim]
        memory = self.transformer(x)
        return memory

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=6, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=100)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, memory):
        # x shape: [B, T]
        x = self.embedding(x)  # [B, T, d_model]
        x = self.pos_encoding(x)
        mask = self.generate_square_subsequent_mask(x.size(1), x.device)
        x = self.transformer(x, memory, mask)
        x = self.fc_out(x)
        return x
    
    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).transpose(0, 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, device):
        super().__init__()
        self.in_shape = in_channels
        self.device = device
        self.vocab_size = out_shape[0]
        
        # Encoder
        self.encoder = ViTEncoder(in_shape[1], emb_dim=768)
        
        # Decoder
        self.decoder = TransformerDecoder(self.vocab_size, d_model=768)
        
    def forward(self, images, captions=None, hidden_state=None):
        if captions is not None:
            memory = self.encoder(images)
            embedded = self.decoder.embedding(captions)
            embedded = embedded + self.decoder.pos_encoding(embedded)
            mask = self.decoder.generate_square_subsequent_mask(embedded.size(1), self.device)
            out = self.decoder.transformer(out, memory, mask)
            logits = self.decoder.fc_out(out)
            return logits, None
        else:
            # Placeholder for the else block
            return None, None

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999))

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images, captions = images.to(self.device), captions.to(self.device)
            out = self.forward(images, captions)
            logits = out[0] if isinstance(out, tuple) else out
            if logits is None:
                continue
            tgt = (captions[:,0,:] if captions.ndim==3 else captions)[:,1:]
            loss = self.criteria[0](logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

            raise NotImplementedError

def supported_hyperparameters():
    return {'lr','momentum'}