def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
              ...

def supported_hyperparameters():
    return {'lr','momentum'}
