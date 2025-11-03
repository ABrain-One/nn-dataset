import torch
import torch.nn as nn
import math

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape[0]
        self.embed_size = prm['embed_size'] if 'embed_size' in prm else 768
        self.hidden_size = self.embed_size
        self.num_layers = prm['num_layers'] if 'num_layers' in prm else 6
        self.nhead = prm['nhead'] if 'nhead' in prm else 8
        
        # Build encoder
        self.encoder = self.make_encoder(in_shape[1], self.embed_size)
        
        # Build decoder
        self.decoder = self.make_transformer_decoder(vocab_size=self.vocab_size, d_model=self.embed_size, num_layers=self.num_layers, nhead=self.nhead)
        
        # Define embedding layer
        self.embed_captions = nn.Embedding(self.vocab_size, self.embed_size)
        
        # Define output layer
        self.fc_out = nn.Linear(self.embed_size, self.vocab_size)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(self.embed_size)
        
        # Initialize hidden state
        self.hidden_state_size = (self.embed_size,)
        self.hidden_state = None
        
        self.to(device)
    
    def make_encoder(self, in_channels, out_channels):
        encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, out_channels)
        )
        return encoder
    
    def make_transformer_decoder(self, vocab_size, d_model, num_layers, nhead):
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        return decoder
    
    def init_hidden(self, batch, device):
        return torch.zeros((batch, self.embed_size), device=device)
    
    def forward(self, images, captions=None, hidden_state=None):
        # First, get memory from images
        memory = self.encoder(images)
        
        # If captions are provided, then we use the decoder and return the logits and hidden_state
        if captions is not None:
            # Embed the captions
            embedded = self.embed_captions(captions)
            embedded = self.pos_enc(embedded)
            
            # Pass through transformer decoder
            output = self.decoder(embedded, memory)
            
            # Project to vocabulary space
            logits = self.fc_out(output)
            
            # If hidden_state is None, initialize it
            if hidden_state is None:
                hidden_state = self.init_hidden(images.size(0), images.device)
                
            return logits, hidden_state
        
        else:
            # Inference mode: return the memory and an initial hidden state if not provided
            if hidden_state is None:
                hidden_state = self.init_hidden(images.size(0), images.device)
            return memory, hidden_state
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999))

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images, captions = images.to(self.device), captions.to(self.device)
            logits = None
            if hasattr(self, 'forward'):
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
        # x is expected to have shape [seq, batch, d_model]
        seq_len, batch, _ = x.size()
        pe = self.pe[:seq_len, :]
        x = x + pe
        return self.dropout(x)

def supported_hyperparameters():
    return {'lr', 'momentum'}