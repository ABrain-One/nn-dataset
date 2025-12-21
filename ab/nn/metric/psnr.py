import torch
import math

# --- 1. Define Normalization Constants ---
# Max and Min PSNR as specified by your teacher (48 dB and 0 dB)
MAX_PSNR_DB = 48.0
MIN_PSNR_DB = 0.0

class Net:
    """
    A stateful metric that calculates the Peak Signal-to-Noise Ratio (PSNR) 
    and returns the score normalized to the range [0, 1].
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the internal state for a new evaluation.
        """
        self._total_mse = 0.0
        self._total_samples = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Updates the state with the results from a new batch.
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
        Computes the raw PSNR, normalizes it to [0, 1], and returns the
        normalized score.
        """
        if self._total_samples == 0:
            # If no samples, return a neutral score (e.g., normalized 0.5)
            return 0.5

        if self._total_mse == 0:
            # If MSE is 0, PSNR is infinite, return the max normalized score
            return 1.0

        mean_mse = self._total_mse / self._total_samples

        max_pixel = 1.0

        # 1. Calculate the Raw PSNR (dB)
        raw_psnr_db = 20 * math.log10(max_pixel / math.sqrt(mean_mse))

        # --- 2. Normalize and Clip the PSNR Score to [0, 1] ---

        # Clip the raw score to the defined range [0, 48]
        psnr_clipped = max(MIN_PSNR_DB, min(MAX_PSNR_DB, raw_psnr_db))

        # Apply normalization formula: (x - min) / (max - min)
        # Note: (MAX_PSNR_DB - MIN_PSNR_DB) is just 48.0 in this case
        normalized_score = (psnr_clipped - MIN_PSNR_DB) / (MAX_PSNR_DB - MIN_PSNR_DB)

        return normalized_score

def create_metric(out_shape=None):
    """
    Factory function required by the training framework to create the metric instance.
    """
    return Net()
