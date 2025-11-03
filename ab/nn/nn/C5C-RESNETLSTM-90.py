import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.out_dim = out_shape[0] if isinstance(out_shape, (tuple, list)) else int(out_shape)
        # We'll define a simple encoder and decoder for demonstration.
        # Encoder: a single convolution
        self.cnn = nn.Conv2d(self.in_channels, 64, 3)
        # Decoder: a simple LSTM
        self.rnn = nn.LSTM(64, 128, 1)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999)
        )


    def learn(self, train_data):
        # This is a placeholder. In the original, it might return a function for learning.
        pass

    def forward(self, images, captions=None, hidden_state=None):
        # images: [B, C, H, W]
        # captions: [B, L] or [B, L, V] (vocabulary size)
        # hidden_state: [B, hidden_size] or None

        # First, process the images through the encoder.
        features = self.cnn(images)  # [B, 64, H-2, W-2]
        # Then, we need to flatten the features to [B, 64, num_spatial]
        # But the decoder expects [B, num_regions, feature_dim]

        # We'll use a global average pooling to get a single feature vector per image.
        features = F.adaptive_avg_pool2d(features, output_size=1)  # [B,64,1,1]
        features = features.view(features.size(0), -1)  # [B,64]

        # The decoder is an LSTM that takes a sequence of inputs and produces a sequence of outputs.
        # We'll use the single feature vector as the initial state for the LSTM.

        # If captions are provided, we are in training mode.
        if captions is not None:
            # We'll convert the captions to embeddings.
            # captions is [B, L] (if out_shape is (vocab_size,)) or [B, L, V] (if out_shape is (vocab_size,))
            # Let's assume the captions are given as [B, L] and the vocabulary size is self.out_dim.
            # We'll define an embedding layer if not present.
            if not hasattr(self, 'embedding'):
                self.embedding = nn.Embedding(self.out_dim, 64)

            captions = captions.long()
            captions = self.embedding(captions)  # [B, L, 64]

            # Now, we have to run the LSTM on the captions.
            # The LSTM expects input of shape [B, L, feature_dim] and hidden state [B, L, hidden_size]
            # But our hidden_state is [B, 1, 128] (from the previous state) and the input is [B, L, 64]

            # If hidden_state is None, initialize it.
            if hidden_state is None:
                hidden_state = self.hidden_state_init(images.size(0))

            captions_embedded = captions.permute(1, 0, 2)  # [L, B, 64]
            output, hidden_state = self.rnn(captions_embedded, hidden_state)
            output = output.permute(1, 0, 2)  # [B, L, 128]

            # Then, project the output to the vocabulary size.
            output = output[:, :, :self.out_dim]  # [B, L, vocab_size]
            output = output.permute(0, 2, 1)  # [B, vocab_size, L]

            return output, hidden_state

        else:
            # We are to generate captions.
            # We'll use the single feature vector to start the LSTM.
            # But the LSTM expects a sequence.

            # We'll start with a SOS token.
            if not hasattr(self, 'sos_token'):
                self.sos_token = torch.tensor([0]).long().to(self.device)

            # Create a batch of SOS tokens.
            batch_size = images.size(0)
            captions = captions.new_full((batch_size, 1), self.sos_token.item())

            # Then, we'll run the LSTM until we generate an EOS token.
            # We'll define an EOS token.
            if not hasattr(self, 'eos_token'):
                self.eos_token = torch.tensor([1]).long().to(self.device)

            # We'll run for a fixed number of steps.
            max_length = 50
            for i in range(max_length):
                # Convert the last token to embedding.
                last_token = captions[:, -1]
                last_token_embed = self.embedding(last_token)  # [B, 64]

                # Now, we have to expand it to [B, 1, 64] and then run the LSTM.
                # But the LSTM expects a sequence of embeddings.

                # We'll create a tensor of embeddings for the entire sequence so far.
                captions_embed = self.embedding(captions)  # [B, i+1, 64]

                # Then, run the LSTM on the entire sequence so far.
                output, hidden_state = self.rnn(captions_embed, hidden_state)

                # Then, project to vocabulary.
                next_token = output[:, -1, :]  # [B, vocab_size]
                next_token = next_token.argmax(dim=-1)  # [B]

                # Append to captions.
                captions = torch.cat([captions, next_token.unsqueeze(1)], dim=1)

                # If we have an EOS token, break.
                if next_token.item() == self.eos_token.item():
                    break

            return captions, hidden_state

    def hidden_state_init(self, batch_size):
        # Initialize hidden state for the LSTM.
        # The LSTM expects hidden state [h, c] each of shape [num_layers, batch_size, hidden_size]
        # We have num_layers=1 and hidden_size=128.
        return (torch.zeros(1, batch_size, 128).to(self.device), torch.zeros(1, batch_size, 128).to(self.device))

    def supported_hyperparameters():
    return {'lr','momentum'}



# --- auto-closed by AlterCaptionNN ---