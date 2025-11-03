import torch.nn as nn

class Net(nn.Module):
    def __init__(self, vocab_size, hidden_size=640, num_attn_heads=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_attn_heads, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def train_setup(self, hparams):
        # Set up the model for training
        pass

    def learn(self, batch, hparams):
        # Training step
        pass

    def forward(self, inputs, hidden_state, memory):
        embedded = self.embedding(inputs)  # [B, T-1, hidden_size]
        output_lstm, new_hidden_state = self.lstm(embedded, hidden_state)  # [B, T-1, hidden_size]
        expanded_memory = memory.expand(memory.size(0), embedded.size(1), -1)  # [B, T-1, hidden_size]
        attn_output, attn_weights = self.attention(output_lstm, expanded_memory, expanded_memory)
        output = self.fc_out(attn_output.transpose(0, 1))  # [T-1, B, vocab_size]
        return output, new_hidden_state

def supported_hyperparameters():
    return {'lr','momentum'}