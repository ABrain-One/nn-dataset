import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from functools import reduce
import operator


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    # KL(N(mu_q, sig_q^2) || N(mu_p, sig_p^2)), summed over params
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1
                + (sig_q / sig_p).pow(2)
                + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


class ModuleWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    # NOTE: Net overrides forward; this is left here for modules that extend ModuleWrapper directly.
    def forward(self, x):
        for module in self.children():
            x = module(x)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return x, kl


class FlattenLayer(ModuleWrapper):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # keep batch dim, flatten the rest
        return x.view(x.size(0), -1)


class KL_Layer(ModuleWrapper):
    def __init__(self):
        super().__init__()
        self.prior_mu = torch.tensor(0.0)
        self.prior_sig = torch.tensor(1.0)
        # Create on default device; .to(device) later will move them.
        self.posterior_mu = Parameter(torch.tensor([0.0], requires_grad=True))
        self.posterior_sig = Parameter(torch.tensor([1.0], requires_grad=True))

    def kl_loss(self):
        return calculate_kl(self.posterior_mu, self.posterior_sig, self.prior_mu, self.prior_sig)


def _shape_to_tuple(shape):
    """Flattens nested (list/tuple of ints) into a flat tuple of ints."""
    if isinstance(shape, int):
        return (shape,)
    if isinstance(shape, (list, tuple)):
        out = []
        for s in shape:
            out += list(_shape_to_tuple(s))
        return tuple(out)
    raise TypeError(f"Unsupported out_shape type: {type(shape)}")


def _numel_from_shape_tuple(shape_tuple):
    return reduce(operator.mul, shape_tuple, 1)


class Net(ModuleWrapper):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        # ---- API aliases (common across your stack) ----
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3

        # For compatibility with earlier patterns; safe no-ops if not used downstream:
        self.vocab_size = int(out_shape[0]) if isinstance(out_shape, (tuple, list)) else int(out_shape)
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # Save training params
        self.prm = prm

        # Output shape handling
        self._out_size = _shape_to_tuple(out_shape)
        out_features = _numel_from_shape_tuple(self._out_size)

        # Minimal network: flatten -> linear -> reshape to out_shape
        self.flatten = FlattenLayer()
        # LazyLinear infers in_features on first forward pass
        self.linear = nn.LazyLinear(out_features)

        # Optional KL collector layer so KL shows up in self.modules()
        self.kl_gate = KL_Layer()

        # Training bits (filled in train_setup)
        self.criteria = None
        self.optimizer = None
        self.recon_criterion = nn.MSELoss()

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get('lr', 1e-3)),
            betas=(float(prm.get('momentum', 0.9)), 0.999),
        )

    def learn(self, train_data):
        assert self.optimizer is not None, "Call train_setup(prm) before learn()."
        self.train()
        for epoch in range(int(self.prm.get('epochs', 1))):
            for x, y in train_data:
                x = x.to(self.device)
                y = y.to(self.device)

                output, kl_loss = self(x)

                # If target shape doesn't include batch, broadcast y
                if y.dim() == output.dim() - 1:
                    y = y.view(output.size(0), *y.shape)

                recon = self.recon_criterion(output, y)
                total = recon + kl_loss

                self.optimizer.zero_grad()
                total.backward()
                self.optimizer.step()

    def forward(self, x):
        x = x.to(self.device)

        z = self.flatten(x)       # [B, N]
        z = self.linear(z)        # [B, prod(out_shape)]
        output = z.view(z.size(0), *self._out_size)

        # Shape check (ignore batch dim)
        assert tuple(output.shape[1:]) == self._out_size, \
            f"Output shape {tuple(output.shape[1:])} != expected {self._out_size}"

        # Accumulate KL from any submodule that defines kl_loss()
        kl_loss = 0.0
        for m in self.modules():
            if hasattr(m, 'kl_loss'):
                kl_loss = kl_loss + m.kl_loss()

        return output, kl_loss


def supported_hyperparameters():
    return {'lr', 'momentum'}
