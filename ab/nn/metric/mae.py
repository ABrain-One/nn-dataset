# MAE metric: normalized to 0-1 accuracy score (1.0 = perfect, 0.0 = MAE >= threshold)
import torch


class Net:
    def __init__(self, max_mae_threshold: float = 20.0):
        self.name = "mae"
        self.max_mae_threshold = max_mae_threshold
        self.reset()

    def reset(self):
        self._total_abs_error = 0.0
        self._total_samples = 0

    # Accumulate absolute errors from a batch
    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = outputs.view(-1)
        labels = labels.view(-1)
        abs_errors = torch.abs(outputs - labels)
        self._total_abs_error += abs_errors.sum().item()
        self._total_samples += outputs.size(0)

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        self.update(outputs, labels)
        return self._total_abs_error, self._total_samples

    def result(self) -> float:
        return self.compute()

    # Returns normalized accuracy: 1.0 - (MAE / threshold)
    def compute(self) -> float:
        if self._total_samples == 0:
            return 0.0
        mae_years = self._total_abs_error / self._total_samples
        return max(0.0, 1.0 - (mae_years / self.max_mae_threshold))

    # Raw MAE in years
    def get_mae(self) -> float:
        if self._total_samples == 0:
            return float('inf')
        return self._total_abs_error / self._total_samples


# Framework factory function
def create_metric(out_shape=None):
    return Net(max_mae_threshold=20.0)
