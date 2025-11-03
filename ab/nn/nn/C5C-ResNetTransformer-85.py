import torch.nn as nn

def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, vocab_size, hidden_size=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def train_setup(self, hparams):
        pass

    def learn(self, images, captions, hidden_state=None):
        pass

    def forward(self, images, captions=None, hidden_state=None):
        if captions is not None:
            # Teacher forcing
            pass
        else:
            # Beam search
            pass