class CNN_Encoder(nn.Module):
        def __init__(self, in_channels, hidden_dim):
            super().__init__()
            self.stem = nn.Sequential(...)
            self.project = nn.Linear(...)

        def forward(self, x):
            # Process through stem and then project the entire feature map to hidden_dim and reshape to sequence
            ...

def supported_hyperparameters():
    return {'lr','momentum'}
