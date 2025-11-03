import torch
import torch.nn as nn
import torch.nn.functional as F
class Conv2dNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = norm_layer(out_channels)
        self.activation = activation_layer()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        pe = torch.zeros(max_len, 1, d_model)
        pe[0, 0, 0] = 1
        for pos in range(1, max_len):
            pe[pos, 0, 0: d_model//2] = torch.sin(pos / 10000.0**(2*i/d_model) for i in range(0, d_model//2))
            pe[pos, 0, d_model//2: d_model] = torch.cos(pos / 10000.0**(2*i/d_model) for i in range(0, d_model//2))
        pe.requires_grad = False
        self.pe = nn.Parameter(pe)
    
    def forward(self, x):
        _, seq_len, _ = x.size()
        pe = self.pe[:seq_len]
        x = x + pe
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=6, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, batch_first=True),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, tgt, memory, hidden_state=None):
        embedded = self.embedding(tgt)
        embedded = self.pos_encoding(embedded)
        seq_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool) * -float('inf'), diagonal=1).to(tgt.device)
        out = self.transformer(embedded, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(out)
        return logits, None

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.vocab_size = int(out_shape[0])
        self.device = device
        
        # Encoder: CNN backbone
        self.stem = Conv2dNormActivation(3, 32, kernel_size=3, stride=2, padding=1)
        self.blocks = nn.Sequential(
            Conv2dNormActivation(32, 64, kernel_size=3, stride=2, padding=1),
            Conv2dNormActivation(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2dNormActivation(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2dNormActivation(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2dNormActivation(512, 768, kernel_size=3, stride=2, padding=1)
        )
        self.encoder = nn.Sequential(self.stem, *self.blocks)
        
        # Decoder: Transformer decoder
        self.decoder = TransformerDecoder(self.vocab_size, d_model=768)
        
    def init_zero_hidden(self, batch: int, device: torch.device):
        return None
        
    def train_setup(self, prm):
        pass
        
    def learn(self, images, captions):
        # Teacher forcing
        memory = self.encoder(images)
        tgt = captions[:, :-1]
        logits, _ = self.decoder(tgt, memory)
        return logits, None
        
    def forward(self, images, captions, hidden_state=None):
        # Shape assertions
        assert images.dim() == 4, "Input images must be 4D"
        assert captions.dim() == 2, "Input captions must be 2D"
        
        # If we are to generate captions without teacher forcing, we use the decoder in an autoregressive manner
        if captions is None:
            # We are to generate captions
            memory = self.encoder(images)
            # Initialize the decoder with the first token being <start>
            start_token = torch.ones(len(images), 1, dtype=torch.long, device=self.device) * self.word2idx.get('<start>', 0)
            tgt = start_token
            
            # Generate captions one token at a time
            for i in range(self.max_length):
                logits, hidden_state = self.decoder(tgt, memory, hidden_state)
                next_token = F.softmax(logits, dim=-1).multinomial(1)
                tgt = torch.cat([tgt, next_token], dim=1)
                
            return tgt[:, 1:], hidden_state
        
        else:
            # Teacher forcing
            memory = self.encoder(images)
            logits, _ = self.decoder(captions, memory)
            return logits, None

def supported_hyperparameters():
    return {'lr', 'momentum'}