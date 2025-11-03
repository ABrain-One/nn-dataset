import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class ViTEncoder(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Linear(3 * patch_size * patch_size, embed_dim)

    def forward(self, images):
        B, C, H, W = images.shape
        patches = F.unfold(images, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2)
        projected = self.proj(patches)
        return projected

class Decoder(nn.Module):
    def __init__(self, embed_dim, num_layers=6, nhead=8, vocab_size=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        decoder_layers = nn.TransformerDecoderLayer(embed_dim, nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, inputs, memory):
        embedded = self.embedding(inputs)
        embedded = self.pos_encoder(embedded)
        mask = torch.triu(torch.ones((inputs.shape[1], inputs.shape[1])), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        out = self.transformer_decoder(tgt=embedded, memory=memory, tgt_mask=mask)
        logits = self.fc_out(out)
        return logits, None

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        
        # Define encoder and decoder
        self.encoder = ViTEncoder()
        self.decoder = Decoder(embed_dim=768, vocab_size=out_shape[0])
        
    def train_setup(self, optimizer, hyperparams):
        pass

    def learn(self, images, captions, hidden_state=None):
        # Process captions
        if captions.ndim == 3:
            caps = captions[:,0,:].long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
        else:
            caps = captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
        # Get memory from encoder
        memory = self.encoder(images)
        
        # Forward through decoder
        logits, _ = self.decoder(inputs, memory)
        
        # Calculate loss
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
        loss = F.cross_entropy(logits, targets, reduction='mean')
        
        return loss

    def forward(self, images, captions=None, hidden_state=None):
        # If captions are provided, use them for teacher forcing
        if captions is not None:
            if captions.ndim == 3:
                caps = captions[:,0,:].long().to(self.device)
                inputs = caps[:, :-1]
                targets = caps[:, 1:]
            else:
                caps = captions.long().to(self.device)
                inputs = caps[:, :-1]
                targets = caps[:, 1:]
                
            memory = self.encoder(images)
            logits, hidden_state = self.decoder(inputs, memory)
            return logits, hidden_state
        
        # Otherwise, return initial hidden state
        return None, None

def supported_hyperparameters():
    return {'lr','momentum'}


from torch import F