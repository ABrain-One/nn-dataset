import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0))) / d_model)
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.view(1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be [B, S, d_model]
        seq_len = x.size(1)
        x = x[:seq_len, :, :]
        x = torch.cat((x, self.pe[:seq_len, :, :]), dim=1)
        return self.dropout(x)

class Net(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(Net, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def supported_hyperparameters():
    return {'lr','momentum'}


    def learn(self, inputs: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # This is a placeholder for any learning-related setup
        pass

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        memory = self.encoder(images)
        if captions is not None:
            inputs = captions[:, :-1]
            logits, hidden_state = self.decoder(inputs, hidden_state, memory)
            return logits, hidden_state
        else:
            # During inference, we need to generate captions
            # This is a placeholder for the inference logic
            return None, None

class CNNEncoder(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, output_channels, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.relu(self.bn5(self.conv5(x)))
        return x

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 768):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.transformer_decoder = nn.TransformerDecoderLayer(hidden_dim, num_heads=8, batch_first=True)
        self.transformer = nn.TransformerDecoder(self.transformer_decoder, num_layers=6)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs: torch.Tensor, hidden_state: Optional[torch.Tensor] = None, memory: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        embedded = self.embedding(inputs)
        embedded = self.pos_encoding(embedded)
        seq_length = inputs.size(1)
        tgt_mask = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool), diagonal=1)
        tgt_mask = (1 - tgt_mask) * -10000.0
        out = self.transformer(embedded, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(out)
        return logits, None

# Example usage
if __name__ == "__main__":
    # Create a dummy model for testing
    encoder = CNNEncoder(input_channels=3, output_channels=768)
    decoder = DecoderRNN(vocab_size=10000)
    model = Net(encoder, decoder)

    # Dummy input
    images = torch.randn(1, 3, 224, 224)
    captions = torch.randint(0, 10000, (1, 10))

    # Forward pass
    logits, hidden_state = model(images, captions)
    print(logits.shape)  # Should be [1, 9, 10000]