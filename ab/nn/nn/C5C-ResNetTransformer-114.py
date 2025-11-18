import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        idx = torch.arange(0, d_model).unsqueeze(0)
        idx = idx.float()
        idx = idx / (idx + 1)[:, 0:64]  # This line was problematic
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[0, 0, 0:d_model//2] = torch.sin(position * div_term)
        pe[1, 0, 0:d_model//2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, 0, :]
        return self.dropout(x)

class CNN_Encoder(nn.Module):
    def __init__(self, in_channels, out_features=768):
        super(CNN_Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, out_features)
        )

    def forward(self, x):
        x = self.features(x)
        return x.unsqueeze(1)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=6, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        embedded = self.embedding(tgt)
        embedded = embedded * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)
        out = self.transformer_decoder(tgt=embedded, memory=memory)
        logits = self.fc_out(out)
        return logits, out

class Net(nn.Module):
    def __init__(self, in_shape, vocab_size, **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.vocab_size = vocab_size
        self.sos_index = 0

        # Encoder
        self.encoder = CNN_Encoder(in_shape[1], out_features=768)

        # Decoder
        self.decoder = TransformerDecoder(vocab_size, d_model=768, num_layers=6, nhead=8)

    def train_setup(self, **kwargs):
        # Set up optimizer and scheduler
        lr = kwargs.get('lr', 0.001)
        momentum = kwargs.get('momentum', 0.9)
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        self.scheduler = None
        if 'scheduler' in kwargs:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

    def learn(self, images, captions=None, **kwargs):
        # Training loop
        if captions is not None:
            # Format captions
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            # Get memory from encoder
            memory = self.encoder(images)
            # Run decoder
            logits, _ = self.decoder(inputs, memory)
            # Calculate loss
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=0)
            return loss

    def forward(self, images, captions=None, hidden_state=None):
        # Main forward function
        if captions is not None:
            # Training mode
            memory = self.encoder(images)
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            logits, _ = self.decoder(inputs, memory)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=0)
            return loss, logits, hidden_state
        else:
            # Generation mode
            if hidden_state is None:
                # Initialize hidden_state
                sos_embedding = self.decoder.embedding(torch.ones(images.size(0)))
                sos_embedding = sos_embedding * math.sqrt(self.decoder.d_model)
                sos_embedding = self.decoder.pos_encoder(sos_embedding)
                hidden_state = sos_embedding
            # Run decoder for generation
            logits, hidden_state = self.decoder(captions, hidden_state)
            return logits, hidden_state

def supported_hyperparameters():
    return {'lr', 'momentum'}