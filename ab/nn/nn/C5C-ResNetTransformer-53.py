class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]] = None
        dropout: float = prm['dropout']
        stochastic_depth_prob: float = 0.2
        num_classes: int = out_shape[0]
        norm_layer: Optional[Callable[..., nn.Module]] = None

        # ... (rest of the code is missing, but we are to fix the error)

        self.encoder = nn.Identity()
        self.rnn = nn.Identity()

    def train_setup(self, **kwargs):
        pass

    def learn(self, **kwargs):
        pass

    def forward(self, images, captions=None, hidden_state=None):
        if captions is not None:
            tgt_input = captions[:, :-1]
            memory = self.encoder(images)
            return self.decoder(tgt_input, memory)
        else:
            return self.batch_beam_search(images)

def supported_hyperparameters():
    return {'lr','momentum'}