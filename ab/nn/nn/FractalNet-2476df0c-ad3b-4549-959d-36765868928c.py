
import torch
import torch.nn as nn
import numpy as np


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        dropout_prob = prm['dropout']
        self.num_columns = 4  # Randomly changed to 4
        self.glob_drop_ratio = 0.6  # Randomly changed to 0.6
        self.loc_drop_prob = 0.16  # Randomly changed to 0.16
        self.num_columns = 4  # Ensuring consistency
        self.glob_num_columns = np.random.randint(0, self.num_columns, size=(1,))

        N = 5
        channels = [64 * (2 ** (i if i != 4 else i - 1)) for i in range(N)]
        dropout_probs = [min(0.5, dropout_prob + i * 0.1) for i in range(N)]

        self.features = nn.Sequential()
        in_channels = in_shape[1]
        for i, out_channels in enumerate(channels):
            self.features.add_module(f"unit{i + 1}", FractalUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                num_columns=self.num_columns,
                loc_drop_prob=self.loc_drop_prob,
                dropout_prob=dropout_probs[i]))
            in_channels = out_channels

        # Compute the correct input size for the linear layer dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *in_shape[1:])
            dummy_output = self.features(dummy_input)
            in_features = dummy_output.view(1, -1).shape[1]

        self.output = nn.Linear(in_features=in_features, out_features=out_shape[0])
        self._init_params()

    def _init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)  # Ensure input tensor is float32
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # Correct feature flattening
        x = self.output(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device).to(torch.float32), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()


class FractalUnit(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.block = FractalBlock(in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.block(x)
        x = self.pool(x)
        return x


class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.num_columns = num_columns
        self.loc_drop_prob = loc_drop_prob
        self.blocks = nn.Sequential()
        depth = 2 ** (num_columns - 1)  # Changed from self.num_columns to 4
        for i in range(depth):
            level_block = nn.Sequential()
            for j in range(num_columns):
                if (i + 1) % (2 ** j) == 0:
                    in_channels_ij = in_channels if (i + 1 == 2 ** j) else out_channels
                    level_block.add_module(f"subblock{j + 1}", drop_conv3x3_block(
                        in_channels=in_channels_ij,
                        out_channels=out_channels,
                        dropout_prob=dropout_prob))
            self.blocks.add_module(f"block{i + 1}", level_block)

    def forward(self, x):
        outs = [x] * self.num_columns
        for level_block in self.blocks:
            outs_i = [block(inputs) for block, inputs in zip(level_block, outs)]
            joined_out = torch.stack(outs_i).mean(dim=0)
            outs[:len(level_block)] = [joined_out] * len(level_block)
        return outs[0]


def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
    )
