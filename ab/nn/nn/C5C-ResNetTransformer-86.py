import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, **kwargs):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.vocab_size = kwargs.get('vocab_size', 10000)

    def train_setup(self, **kwargs):
        pass

    def learn(self, images, captions):
        assert images.dim() == 4
        assert captions.ndim == 2 or captions.ndim == 3
        if captions.ndim == 3:
            caps = captions[:,0,:].long().to(self.device)
        else:
            caps = captions[:,0,:].long().to(self.device)
        inputs = caps[:, :-1]
        targets = caps[:, 1:]
        memory = self.encoder(images)
        logits, hidden_state = self.decoder.rnn(inputs, None, memory)
        assert logits.shape == (images.shape[0], inputs.shape[1], self.vocab_size)
        assert hidden_state.shape == (images.shape[0], self.decoder.hidden_dim)
        return logits, hidden_state

    def forward(self, images, captions=None):
        if captions is None:
            return self.inference(images)
        else:
            return self.learn(images, captions)

    def inference(self, images):
        pass

def supported_hyperparameters():
    return {'lr','momentum'}