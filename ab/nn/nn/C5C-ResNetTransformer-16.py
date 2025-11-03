if captions is not None:
              caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
              inputs = caps[:, :-1]   # Shape [B, T-1]
              logits, hidden_state = self.rnn(inputs, hidden_state, memory)   # Here, hidden_state is None initially in teacher forcing, but we have to return a hidden_state even if it wasn't provided.

def supported_hyperparameters():
    return {'lr','momentum'}
