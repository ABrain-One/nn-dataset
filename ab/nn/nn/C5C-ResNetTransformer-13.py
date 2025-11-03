if captions is not None:
              caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
              inputs = caps[:, :-1]
              targets = caps[:, 1:]
              logits, hidden_state = self.rnn(inputs, None, memory)
              ...

def supported_hyperparameters():
    return {'lr','momentum'}
