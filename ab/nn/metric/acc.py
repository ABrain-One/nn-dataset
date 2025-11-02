import torch
from .base import BaseMetric
from ab.nn.metric.base.utils import flatten_logits_and_labels

class Accuracy(BaseMetric):
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        outputs, targets = flatten_logits_and_labels(outputs, targets)

        _, predicted = torch.max(outputs, dim=1)
        self.correct += (predicted == targets).sum().item()
        self.total   += targets.numel()

    def __call__(self, outputs, targets):
        self.update(outputs, targets)
        return self.correct, self.total
    
    def result(self):
        return self.correct / max(self.total, 1e-8)

# Function to create metric instance
def create_metric(out_shape=None):
    return Accuracy()
