import torch
import torch.nn as nn

def supported_hyperparameters():
    # FIX: ensure the body is indented under the function definition
    return {'lr', 'momentum'}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *args, **kwargs):
        super().__init__()
        self.device = device

    def train_setup(self, prm):
        pass

    def learn(self, train_data):
        pass

    def forward(self, *args, **kwargs):
        pass
