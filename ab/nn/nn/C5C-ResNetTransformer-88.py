import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, :div_term.size(0)] = torch.sin(position * div_term)
        pe[:, 0, div_term.size(0):] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0) if x.dim() == 3 else x.size(1)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=6, nhead=8):
        super(DecoderTransformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2048, batch_first=True),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        tgt = tgt.to(torch.long)
        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.pos_encoder(tgt_embedding)
        memory = memory.transpose(0, 1)  # [S, B, H]
        output = self.transformer(tgt_embedding, memory)
        logits = self.fc_out(output)
        hidden_state = output[:, -1, :]
        return logits, hidden_state

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device

        # Encoder
        input_channels = in_shape[1]
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc_encoder = nn.Linear(128, 768)

        # Decoder
        self.decoder = DecoderTransformer(
            vocab_size=out_shape[0],
            d_model=768,
            num_layers=prm.get('num_layers', 6),
            nhead=prm.get('nhead', 8)
        )

    def train_setup(self, optimizer, lr, momentum):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            if 'momentum' in param_group:
                param_group['momentum'] = momentum

    def learn(self, images, captions=None, hidden_state=None):
        if captions is not None:
            # Training mode
            memory = self.encoder(images)
            tgt = captions[:, :-1]
            targets = captions[:, 1:]
            logits, hidden_state = self.decoder(tgt, memory)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits.transpose(0,1,2), targets)
            return loss
        else:
            # Inference mode
            memory = self.encoder(images)
            # Start with SOS token and create empty tensors for predictions
            # This is a placeholder for beam search or other generation methods
            # We'll return a dummy caption for demonstration
            return torch.randint(0, self.decoder.embedding.num_embeddings, (images.size(0), 15))

    def forward(self, images, captions=None, hidden_state=None):
        if captions is not None:
            assert captions.dim() == 2, "Captions must be 2D tensor"
            assert images.dim() == 4, "Images must be 4D tensor"
            memory = self.encoder(images)
            tgt = captions[:, :-1]
            targets = captions[:, 1:]
            logits, hidden_state = self.decoder(tgt, memory)
            assert logits.shape == (captions.size(0), captions.size(1)-1, self.out_shape[0]), "Logits shape must be [B, T-1, vocab_size]"
            return logits, hidden_state
        else:
            memory = self.encoder(images)
            return self.decoder(None, memory), None

def supported_hyperparameters():
    return {'lr','momentum'}