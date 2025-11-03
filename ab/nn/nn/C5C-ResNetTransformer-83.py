import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, hidden_state=None, memory=None):
        # inputs: [B, T]
        # hidden_state: tuple or None (expected to be [num_layers*b, hidden_dim])
        # memory: [B, 1, hidden_dim]

        # If hidden_state is None, initialize it from memory.
        if hidden_state is None:
            batch_size = inputs.size(0)
            hidden_state = (torch.zeros(1, batch_size, self.hidden_dim), torch.zeros(1, batch_size, self.hidden_dim))
            h0, c0 = hidden_state
            h0[0, :, :] = memory[:, :, 0]
            c0[0, :, :] = h0[0, :, :].clone()
            hidden_state = (h0, c0)
        else:
            # Use provided hidden_state
            pass

        embedded = self.embedding(inputs)
        if hidden_state is None:
            first_embed = embedded[:,0] + memory[:,0]
            extended_embed = torch.cat([first_embed.unsqueeze(1), embedded[:,1:]], dim=1)
            embedded = extended_embed

        output, hidden_state = self.lstm(embedded, hidden_state)
        logits = []
        for i in range(output.size(1)):
            logits.append(self.linear(output[:, i]))
        logits = torch.stack(logits, 1)
        return logits, hidden_state

class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.rnn = LSTMDecoder(params.vocab_size, hidden_dim=768)
        self.criterion = nn.CrossEntropyLoss().to(params.device)

    def train_setup(self, optimizer):
        pass

    def learn(self, inputs, memory, captions=None, caption_lengths=None, use_teacher_forcing=True, beam_size=None, test_mode=False):
        if captions is not None:
            tgt_input = captions[:, :-1]
            logits, hidden_state = self.rnn(tgt_input, None, memory)
            targets = captions[:, 1:]
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return loss, hidden_state
        else:
            pass

    def forward(self, inputs, hidden_state, memory):
        return self.rnn(inputs, hidden_state, memory)

def supported_hyperparameters():
    return {'lr','momentum'}