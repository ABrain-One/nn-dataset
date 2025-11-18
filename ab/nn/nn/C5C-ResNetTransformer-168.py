import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape[0]
        self.hidden_dim = 768

        # Define the encoder
        self.encoder = EncoderCNN(in_shape[1], self.hidden_dim)

        # Define the decoder
        self.rnn = DecoderTransformer(self.vocab_size, self.hidden_dim, 6, 8)

    def init_zero_hidden(self, batch, device):
        return None, None

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=prm.get('ignore_index', 0)).to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'])

    def learn(self, images, captions):
        inputs = captions[:, :-1]
        targets = captions[:, 1:]
        memory = self.encoder(images)
        logits, _ = self.rnn(inputs, None, memory)
        loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return loss

    def forward(self, images, captions=None, hidden_state=None):
        if captions is not None:
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            memory = self.encoder(images)
            logits, hidden_state = self.rnn(inputs, hidden_state, memory)
            return logits, hidden_state
        else:
            memory = self.encoder(images)
            hidden_state = self.init_zero_hidden(images.size(0), self.device)
            return hidden_state, memory

class EncoderCNN(nn.Module):
    def __init__(self, input_channels, output_dim=768):
        super().__init__()
        self.cnn = nn.Sequential(
            # First convolution block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second convolution block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third convolution block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Global average pooling
            nn.AdaptiveAvgPool2d((1,1)),
            # Flatten
            nn.Flatten(),
            # Linear layer to output_dim
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.cnn(x)

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, nhead):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead),
            num_layers
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, hidden_state, memory):
        embedded = self.embedding(inputs)
        embedded = embedded.permute(1, 0, 2)  # [T, B, hidden_size]
        embedded = self.pos_encoding(embedded)
        embedded = embedded.permute(1, 0, 2)   # [B, T, hidden_size]

        # Create a mask for the target (causal mask)
        tgt_mask = self.generate_square_subsequent_mask(embedded.size(1)).to(self.device)

        out = self.transformer(embedded, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(out)
        return logits, hidden_state

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.embed_positions = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x is [B, T, d_model]
        seq_len = x.size(1)
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        return self.dropout(x + self.embed_positions(positions))

def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

def supported_hyperparameters():
    return {'lr','momentum'}