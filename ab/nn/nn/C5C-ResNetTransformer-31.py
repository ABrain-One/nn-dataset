logits, hidden_state = self.rnn(inputs, None, memory)

def supported_hyperparameters():
    return {'lr','momentum'}
