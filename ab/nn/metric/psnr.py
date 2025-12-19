import torch
import math

class Net:
    def __init__(self):
        self.reset()
        self.name = "mai_psnr"

    def reset(self):
        # This clears the memory before a new test starts
        self._total_mse = 0.0
        self._total_samples = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        # 1. Look only at Brightness (Y-channel)
        # We take Red, Green, Blue and mix them into one 'Gray' layer
        y_out = 0.299 * outputs[:,0] + 0.587 * outputs[:,1] + 0.114 * outputs[:,2]
        y_lab = 0.299 * labels[:,0] + 0.587 * labels[:,1] + 0.114 * labels[:,2]

        # 2. Calculate the 'Mistake' (Mean Squared Error)
        mse = torch.mean((y_out - y_lab) ** 2)

        # 3. Add this result to our 'Total' pile
        self._total_mse += mse.item() * outputs.size(0)
        self._total_samples += outputs.size(0)

    def compute(self):
        # This part gives the final answer
        if self._total_samples == 0:
            return 0.0
        
        avg_mse = self._total_mse / self._total_samples
        if avg_mse == 0: return 1.0 # Perfect!

        # Calculate standard PSNR
        raw_psnr = 10 * math.log10(1.0 / avg_mse)

        # Normalize by 48 (The Professor's Rule)
        # This makes sure the result is 0.0 to 1.0
        final_score = min(raw_psnr / 48.0, 1.0)
        
        return final_score
