import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        # Basic aliases the rest of the stack often expects
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.priors = prm  # keep around if needed later

        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3

        # Minimal conv + pool to match the original intent
        self.conv1 = nn.Conv2d(self.in_channels, 6, kernel_size=5, padding=0, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get('lr', 1e-3)),
            betas=(float(prm.get('momentum', 0.9)), 0.999),
        )

    def learn(self, train_data):
        # Placeholder: training loop is dataset/loader specific.
        pass

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.pool1(x)
        return x
