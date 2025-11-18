import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class VisionTransformerEncoder(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=6, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patch = (image_size // patch_size) ** 2

        # Stem
        stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.stem = stem

        # Projection
        self.proj = nn.Conv2d(64, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patch, embed_dim))
        truncage_length = self.n_patch
        nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*mlp_ratio)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        x = self.stem(x)
        x = self.proj(x)
        x = x.reshape(x.size(0), x.size(1), -1)  # [B, embed_dim, n_patch]
        x = x.permute(0, 2, 1)  # [B, n_patch, embed_dim]
        pos = self.pos_embed.repeat(x.size(0), 1, 1)
        x = x + pos
        x = self.transformer(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape[0])
        in_channels = int(in_shape[1])
        self.hidden_dim = 768

        self.encoder = VisionTransformerEncoder(in_chans=in_channels)

        # Define the embedding layer for the decoder
        self.embed = nn.Embedding(self.vocab_size, 768)
        self.embed.padding_idx = 0

        # Define the LSTM decoder
        self.rnn = nn.LSTM(768, 768, 2, batch_first=True)
        self.init_hidden = lambda batch_size: (torch.zeros(2, batch_size, 768).to(self.device),
                                              torch.zeros(2, batch_size, 768).to(self.device))

        # Define the generator
        self.generator = nn.Linear(768, self.vocab_size)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
        self.scheduler = None

    def learn(self, train_data):
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.long().to(self.device)

            # Get memory from encoder
            memory = self.encoder(images)

            # Get input and target
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            # Embed the inputs
            embedded = self.embed(inputs)

            # Initialize hidden state to zeros
            h0, c0 = self.init_hidden(images.size(0)), self.init_hidden(images.size(0))
            output, hidden = self.rnn(embedded, (h0, c0))

            logits = self.generator(output)

            # Calculate loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images, captions=None, hidden_state=None):
        if captions is not None:
            # Training mode
            memory = self.encoder(images)
            inputs = captions[:, :-1]
            embedded = self.embed(inputs)
            output, hidden = self.rnn(embedded, hidden_state)
            logits = self.generator(output)
            return logits, hidden
        else:
            # Evaluation mode
            memory = self.encoder(images)
            # We'll assume the start token is 1
            start_token = torch.tensor([1]).expand(images.size(0), -1).to(self.device)
            embedded = self.embed(start_token)
            output, hidden = self.rnn(embedded, hidden_state)
            logits = self.generator(output)
            return logits, hidden

def supported_hyperparameters():
    return {'lr','momentum'}