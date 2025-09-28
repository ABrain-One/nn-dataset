# Auto-generated single-file for RecurrentBlock
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn

# ---- torchvision.models.optical_flow.raft.ConvGRU ----
class ConvGRU(nn.Module):
    """Convolutional Gru unit."""

    def __init__(self, *, input_size, hidden_size, kernel_size, padding):
        super().__init__()
        self.convz = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.convr = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.convq = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h

# ---- torchvision.models.optical_flow.raft._pass_through_h ----
def _pass_through_h(h, _):
    # Declared here for torchscript
    return h

# ---- RecurrentBlock (target) ----
class RecurrentBlock(nn.Module):
    """Recurrent block, part of the update block.

    Takes the current hidden state and the concatenation of (motion encoder output, context) as input.
    Returns an updated hidden state.
    """

    def __init__(self, *, input_size, hidden_size, kernel_size=((1, 5), (5, 1)), padding=((0, 2), (2, 0))):
        super().__init__()

        if len(kernel_size) != len(padding):
            raise ValueError(
                f"kernel_size should have the same length as padding, instead got len(kernel_size) = {len(kernel_size)} and len(padding) = {len(padding)}"
            )
        if len(kernel_size) not in (1, 2):
            raise ValueError(f"kernel_size should either 1 or 2, instead got {len(kernel_size)}")

        self.convgru1 = ConvGRU(
            input_size=input_size, hidden_size=hidden_size, kernel_size=kernel_size[0], padding=padding[0]
        )
        if len(kernel_size) == 2:
            self.convgru2 = ConvGRU(
                input_size=input_size, hidden_size=hidden_size, kernel_size=kernel_size[1], padding=padding[1]
            )
        else:
            self.convgru2 = _pass_through_h

        self.hidden_size = hidden_size

    def forward(self, h, x):
        h = self.convgru1(h, x)
        h = self.convgru2(h, x)
        return h

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.recurrent_block = RecurrentBlock(input_size=32, hidden_size=32)
        self.classifier = nn.Linear(32, self.num_classes)
        self.hidden_state = None

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        if self.hidden_state is None or self.hidden_state.shape != x.shape:
            self.hidden_state = torch.zeros_like(x)
        self.hidden_state = self.recurrent_block(self.hidden_state.detach(), x)
        x = self.hidden_state.mean(dim=(2, 3))
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, data_roll):
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
