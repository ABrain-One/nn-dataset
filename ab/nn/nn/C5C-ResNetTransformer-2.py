def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        ...
        if captions is not None:
            ...
            logits, hidden_state = self.rnn(inputs, hidden_state, memory)
            return logits, hidden_state

def supported_hyperparameters():
    return {'lr','momentum'}
