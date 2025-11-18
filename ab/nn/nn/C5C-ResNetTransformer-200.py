import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        
        position = torch.arange(max_len).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[:, :max_len, 0: d_model//2 * 2] = torch.sin(position[:, :max_len, 0: d_model//2 * 2] / div_term)
        pe[:, :max_len, d_model//2 * 2:] = torch.cos(position[:, :max_len, d_model//2 * 2:] / div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class ViTEncoder(nn.Module):
    def __init__(self, in_channels, hidden_size=768):
        super().__init__()
        self.final_image_size = 224
        self.patch_size = 16
        self.num_patches = (self.final_image_size // self.patch_size) ** 2
        self.projection = nn.Linear(in_channels * (self.patch_size**2), hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        if x.shape[-1] != self.final_image_size:
            raise ValueError(f"Input image size must be {self.final_image_size}. Got {x.shape[-1]}.")
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size, padding=0)
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        x = self.layer_norm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=6, nhead=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, memory):
        embedded = self.embedding(tgt)
        embedded = self.pos_encoding(embedded)
        out = self.transformer_decoder(embedded, memory)
        out = self.fc_out(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape[0]
        in_channels, img_h, img_w = in_shape
        self.hidden_dim = 768

        self.encoder = ViTEncoder(in_channels, self.hidden_dim)
        self.decoder = Decoder(self.vocab_size, self.hidden_dim)

    def train_setup(self, prm):
        pass

    def learn(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        memory = self.encoder(images)
        if captions is not None:
            embedded = self.decoder.embedding(captions)
            embedded = self.decoder.pos_encoding(embedded)
            out = self.decoder.transformer_decoder(embedded, memory)
            logits = self.decoder.fc_out(out)
            return logits, out
        else:
            raise NotImplementedError()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        memory = self.encoder(images)
        if captions is not None:
            embedded = self.decoder.embedding(captions)
            embedded = self.decoder.pos_encoding(embedded)
            out = self.decoder.transformer_decoder(embedded, memory)
            logits = self.decoder.fc_out(out)
            return logits, out
        else:
            raise NotImplementedError()

def supported_hyperparameters():
    return {'lr','momentum'}