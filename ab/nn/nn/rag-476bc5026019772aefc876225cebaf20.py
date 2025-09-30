# Auto-generated single-file for PositionAwareLayer
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn

# ---- PositionAwareLayer (target) ----
class PositionAwareLayer(nn.Module):

    def __init__(self, dim_model, rnn_layers=2):
        super().__init__()

        self.dim_model = dim_model

        self.rnn = nn.LSTM(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=rnn_layers,
            batch_first=True)

        self.mixer = nn.Sequential(
            nn.Conv2d(
                dim_model, dim_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(
                dim_model, dim_model, kernel_size=3, stride=1, padding=1))

    def forward(self, img_feature):
        n, c, h, w = img_feature.size()

        rnn_input = img_feature.permute(0, 2, 3, 1).contiguous()
        rnn_input = rnn_input.view(n * h, w, c)
        rnn_output, _ = self.rnn(rnn_input)
        rnn_output = rnn_output.view(n, h, w, c)
        rnn_output = rnn_output.permute(0, 3, 1, 2).contiguous()

        out = self.mixer(rnn_output)

        return out

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
        self.pos_aware = PositionAwareLayer(dim_model=32, rnn_layers=1)
        self.classifier = nn.Linear(32, self.num_classes)

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
        x = self.pos_aware(x)
        x = x.mean(dim=(2, 3))
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
