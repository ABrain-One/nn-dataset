import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.prm = prm or {}

        # Robustly infer feature dims from shape specs
        def _last_dim(shape):
            if isinstance(shape, int):
                return int(shape)
            if isinstance(shape, (tuple, list)) and len(shape) > 0:
                return int(shape[-1])
            return int(shape)

        input_dim = _last_dim(in_shape)
        output_dim = _last_dim(out_shape)

        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, output_dim)

        # Will be initialized in train_setup
        self.criteria = None
        self.optimizer = None

    def train_setup(self, prm):
        prm = prm or {}
        self.to(self.device)
        # Use MSE since the forward returns continuous outputs by default here
        self.criteria = (nn.MSELoss().to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get('lr', 1e-3)),
            betas=(float(prm.get('momentum', 0.9)), 0.999),
        )

    def learn(self, train_data):
        if self.optimizer is None or self.criteria is None:
            self.train_setup(self.prm)

        self.train()
        criterion = self.criteria[0]
        for inputs, targets in train_data:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

    def forward(self, x, teacher_forcing=True):
        # Accept any leading dims; apply Linear over the last dim
        x = x.to(self.device)
        in_features = self.linear1.in_features
        assert x.size(-1) == in_features, (
            f"Expected last dim {in_features}, got {x.size(-1)}"
        )
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
