import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        pe = torch.zeros(max_len, 1, d_model)
        pe[0,0,0:6] = torch.arange(0, max_len).unsqueeze(1) % 2048
        for i in range(1, max_len):
            pe[i,0,0] = i % 2048
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch, seq, d_model]
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

def generate_square_subsequent_mask(sz):
    """Generates a Square causal mask of size [sz, sz]"""
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    mask = mask.masked_fill(mask == 0, float(0.0))
    return mask

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=6, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=2048, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        embedded = self.embedding(tgt)
        embedded = self.pos_encoding(embedded)
        tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        memory_mask = None
        out = self.transformer_decoder(
            tgt=embedded,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        return self.fc_out(out)

class Net(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=6, nhead=8):
        super().__init__()
        self.decoder = TransformerDecoder(vocab_size, d_model, num_layers, nhead)

    def train_setup(self, hparams):
        # Set up for training, e.g., learning rate, etc.
        pass

    def learn(self, images, captions):
        # Teacher forcing: pass captions as next input at each step
        memory = self.encoder(images)
        output = self.decoder(captions, memory)
        return output

    def forward(self, images, captions):
        # Shape asserts
        assert images.dim() == 4, "images must be 4D"
        assert captions.dim() == 2, "captions must be 2D"
        # Teacher forcing
        return self.learn(images, captions)

def supported_hyperparameters():
    return {'lr', 'momentum'}