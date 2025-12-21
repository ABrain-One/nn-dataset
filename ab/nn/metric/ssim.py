import torch
import torch.nn.functional as F
import math
from ab.nn.util.ssim_module import SSIM # Import the new module

ssim_calculator = SSIM(channel=3, size_average=False) 

class Net:
    def __init__(self):
        self.reset()
        self._total_ssim = 0.0
        self._total_batches = 0
        self._total_samples = 0 

    def reset(self):
        self._total_ssim = 0.0
        self._total_batches = 0
        self._total_samples = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        if outputs.dim() == 3:
            outputs = outputs.unsqueeze(0)
            labels = labels.unsqueeze(0)
        
        device = outputs.device
        
        ssim_values = ssim_calculator(outputs, labels.to(device)).to(device)
        
        self._total_ssim += ssim_values.sum().item()
        self._total_samples += outputs.size(0) 
        self._total_batches += 1

    def __call__(self, outputs, labels):
        self.update(outputs, labels)

    def result(self):
        if self._total_samples == 0:
            return 0.0

        average_ssim = self._total_ssim / self._total_samples
        return average_ssim

def create_metric(out_shape=None):
    return Net()
