def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
            images = images.to(self.device, dtype=torch.float32)
            memory = self.encoder(images)
            if captions is not None:
                caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
                inputs = caps[:, :-1]
                logits, hidden_state = self.rnn(inputs, hidden_state, memory)
                assert logits.shape == inputs.shape[1]   # This line is confusing: 
                assert logits.shape[-1] == self.vocab_size
                return logits, hidden_state
            else:
                raise NotImplementedError()

def supported_hyperparameters():
    return {'lr','momentum'}
