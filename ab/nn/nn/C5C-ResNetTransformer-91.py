import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random

class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.device = params['device']
        self.vocab_size = params['vocab_size']
        self.embedder = nn.Embedding(self.vocab_size, 768)
        self.rnn = nn.LSTM(768, 768, batch_first=True)
        self.hidden_dim = 768

        # Encoder backbone
        self.encoder = nn.Identity()

        # Decoder backbone
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=768, nhead=8),
            num_layers=6,
            batch_first=True
        )

        # Final projection for the transformer output
        self.final_proj = nn.Linear(768, self.vocab_size)

    def train_setup(self, params):
        pass

    def learn(self, images, captions):
        # Convert captions to tensor if not already
        if isinstance(captions, list):
            captions = torch.tensor(captions, dtype=torch.long, device=self.device)
        else:
            captions = captions.to(self.device)

        # If captions is 3D, then we assume it's [B, T, V] and we take the first frame and then the rest
        # But the prompt says image captioning, so we assume captions is 2D: [B, T]
        # Let's assume captions is 2D.

        # We'll call forward with captions and images, and then compute the loss.
        # Forward returns (logits, hidden_state)
        logits, hidden_state = self.forward(images, captions)

        # Then compute loss
        loss = self.criterion(logits.reshape(-1, self.vocab_size), captions[:, 1:].reshape(-1))

        return loss

    def forward(self, images, captions=None, hidden_state=None):
        # Process images through encoder
        memory = self.encoder(images)
        memory = memory.permute(0, 2, 3, 1).contiguous().view(memory.size(0), -1, memory.size(1))
        
        # Process captions if provided
        if captions is not None:
            # Convert captions to embeddings
            embedded = self.embedder(captions)
            # Pass through transformer decoder
            output = self.transformer_decoder(embedded, memory)
            # Project to vocabulary space
            logits = self.final_proj(output)
            # Return logits and the last hidden state
            return logits, output[:, -1]

        # If captions is None, we are in inference mode
        # We need to return a hidden state for the decoder
        # For inference, we can return the initial hidden state
        if hidden_state is None:
            hidden_state = torch.zeros((self.rnn.num_layers * self.rnn.hidden_size, images.size(0)), device=self.device)
        else:
            hidden_state = hidden_state.to(self.device)

        # We'll return a dummy output for inference
        return None, hidden_state

def supported_hyperparameters():
    return {'lr','momentum'}