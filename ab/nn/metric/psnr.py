import torch
import math

class Net:
    """
    A stateful metric that calculates the Peak Signal-to-Noise Ratio (PSNR) 
    over an entire dataset by accumulating results from each batch.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the internal state for a new evaluation. This is called at the
        beginning of the evaluation loop.
        """
        self._total_mse = 0.0
        self._total_samples = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Updates the state with the results from a new batch. This is called
        for every batch in the evaluation loop.
        """
        device = outputs.device
        outputs = outputs.to(device).float()
        labels = labels.to(device).float()
        
        batch_mse = torch.sum((outputs - labels) ** 2)
        
        self._total_mse += batch_mse.item()
        self._total_samples += outputs.numel()

    def __call__(self, outputs, labels):
        """
        This method is called for each batch. It should only update the
        internal state and NOT return a value to avoid framework errors.
        """
        self.update(outputs, labels)

    def result(self):
        """
        Computes and returns the final PSNR from the accumulated state.
        This is called once at the end of the evaluation loop.
        """
        if self._total_samples == 0 or self._total_mse == 0:
            return 100.0

        mean_mse = self._total_mse / self._total_samples
        
        max_pixel = 1.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mean_mse))
        
        return psnr

def create_metric(out_shape=None):
    """
    Factory function required by the training framework to create the metric instance.
    """
    return Net()
