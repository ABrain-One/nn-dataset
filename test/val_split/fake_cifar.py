"""A tiny in-memory stand-in for the cifar-10 loader (no download): 10 classes, 3x32x32 images whose mean encodes the
class, so a small network learns it in one epoch. Datasets carry .targets, .transform and num_workers = 0 like the
torchvision datasets the true-test loaders return."""
import torch
from torch.utils.data import Dataset

MINIMUM_ACCURACY = 0.0
N_TRAIN, N_TEST, N_CLASSES = 400, 100, 10


class FakeVision(Dataset):
    def __init__(self, n, seed, transform=None):
        g = torch.Generator().manual_seed(seed)
        self.targets = [i % N_CLASSES for i in range(n)]
        base = torch.randn(n, 3, 32, 32, generator=g) * 0.5
        self.data = base + torch.tensor(self.targets, dtype=torch.float32).view(n, 1, 1, 1) / 3.0
        self.transform = transform
        self.num_workers = 0

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        x = self.data[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.targets[i]


def loader(transform_fn, task):
    transform = transform_fn(((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return (N_CLASSES,), MINIMUM_ACCURACY, FakeVision(N_TRAIN, 1, transform), FakeVision(N_TEST, 2, transform)


def install():
    """Register this module as ab.nn.loader.cifar-10 (load_dataset imports loaders by name)."""
    import sys
    import ab.nn.loader  # noqa: F401
    sys.modules["ab.nn.loader.cifar-10"] = sys.modules[__name__]


def prepare_workdir(base):
    """What a run from the repository root has: the working directory on sys.path (nn-dataset imports the model as
    out.nn.tmp.<name> under it) and, when nn-dataset is not pip-installed (a checkout on PYTHONPATH), the metric
    source under <cwd>/ab/nn/metric (nn_mod reads it from the project root)."""
    import shutil
    import sys
    from pathlib import Path
    import ab.nn.metric
    if str(base) not in sys.path:
        sys.path.append(str(base))
    dst = Path(base) / "ab" / "nn" / "metric"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(ab.nn.metric.__file__).parent / "acc.py", dst / "acc.py")


NET_CODE = '''import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.conv = nn.Conv2d(in_shape[1], 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(8 * 16, out_shape[0])

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criteria[0](self(inputs), labels)
            loss.backward()
            self.optimizer.step()

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(torch.relu(self.conv(x))), 1))
'''

PRM = {'lr': 0.05, 'momentum': 0.9, 'batch': 32, 'epoch': 2, 'transform': 'ident', 'dropout': 0.0}
