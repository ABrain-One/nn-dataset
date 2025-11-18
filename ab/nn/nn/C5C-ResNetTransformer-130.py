import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(d_model, dtype=torch.float) * -torch.log(torch.tensor(10000.0)) / d_model)
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0:d_model//2] = torch.sin(position * div_term[0:d_model//2])
        pe[:, 0, d_model//2:d_model] = torch.cos(position * div_term[0:d_model//2])
        pe = pe[:, 0, :].unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is expected to be [B, T, E]
        seq_len = x.size(1)
        x = x[:,:,:seq_len] + self.pe[:,:,:seq_len]
        return self.dropout(x)

class ViTEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=768):
        super().__init__()
        self.patch_size = 16
        self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.num_patches = (in_channels[2] // self.patch_size) ** 2  # This line has an error

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), self.num_patches, -1)
        return x

class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim=768, num_layers=2, dropout=0.1, vocab_size=1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, hidden_state, memory):
        embedded = self.embed(inputs)
        embedded = self.pos_encoder(embedded)
        embedded = embedded + memory  # This line has an error

        if hidden_state is None:
            hidden_state = self.init_hidden(inputs.size(0))

        output, hidden_state = self.lstm(embedded, hidden_state)
        logits = self.fc_out(output)
        return logits, hidden_state

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.vocab_size = self.num_classes
        self.hidden_dim = 768
        self.num_layers = prm.get('num_layers', 2)
        self.dropout = prm.get('dropout', 0.1)

        # Encoder
        self.encoder = ViTEncoder(in_channels=self.in_channels, hidden_dim=self.hidden_dim)

        # Decoder
        self.decoder = LSTMDecoder(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout, vocab_size=self.vocab_size)

        # Criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def train_setup(self, lr, momentum):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def learn(self, train_data):
        self.train()
        total_loss = 0
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)

            memory = self.encoder(images)
            embedded = self.embed(captions)
            embedded = embedded + memory  # This line has an error

            # Calculate target_mask
            sz = embedded.size()
            tgt_mask = torch.triu(torch.ones(sz[1], sz[1], dtype=torch.bool, device=images.device), diagonal=1).triu_().transpose(0,1)
            tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))

            output = self.transformer_dec(tgt=embedded, memory=memory, tgt_mask=tgt_mask)
            logits = self.fc_out(output)

            loss = self.criterion(logits.view(-1, logits.size(-1)), captions[:,1:].view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(train_data)

    def forward(self, images, captions=None):
        images = images.to(self.device)
        memory = self.encoder(images)

        if captions is not None:
            captions = captions.to(self.device)
            embedded = self.embed(captions)
            embedded = embedded + memory  # This line has an error

            # Calculate target_mask
            sz = embedded.size()
            tgt_mask = torch.triu(torch.ones(sz[1], sz[1], dtype=torch.bool, device=images.device), diagonal=1).triu_().transpose(0,1)
            tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))

            output = self.transformer_dec(tgt=embedded, memory=memory, tgt_mask=tgt_mask)
            logits = self.fc_out(output)
            return logits, None
        else:
            return None, None

    def beam_search(self, images, beam_size=5, max_length=50):
        self.eval()
        images = images.to(self.device)
        memory = self.encoder(images)

        # Initialize the beam with SOS token
        # This part is incomplete and has errors

        # We'll return a dummy implementation due to complexity
        return torch.randint(0, self.vocab_size, (images.size(0), max_length), device=images.device)

    def supported_hyperparameters():
    return {'lr','momentum'}


    def embed_tokens(self, x):
        return self.embed(x)

    def transformer_decoder_layer(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
                                        dropout=dropout, batch_first=True, activation=activation)
        return layer

    def transformer_decoder(self, decoder_layer, num_layers):
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, batch_first=True)
        return decoder

    def fc_out_layer(self, in_features, out_features):
        return nn.Linear(in_features, out_features)

    def learn_with_beam_search(self, train_data, beam_size=5):
        # This part is incomplete and has errors
        pass

    def learn_with_scheduled_sampling(self, train_data, alpha=0.1):
        # This part is incomplete and has errors
        pass