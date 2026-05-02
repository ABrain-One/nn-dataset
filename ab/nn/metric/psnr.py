
import torch
import math

class Net:
    """
    PSNR metric for Super Resolution.
    - Computes PSNR on Y-channel (luminance) following standard SR evaluation
    - Normalizes to 0.0-1.0 range using max PSNR of 48 dB
    - Reuses implementation pattern from mai_psnr.py
    """
    def __init__(self, out_shape=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset()

    def reset(self):
        """Clear accumulated metrics before new evaluation"""
        self.total_mse = 0.0
        self.count = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Update metric with batch of predictions and ground truth.
        
        Args:
            outputs: Predicted images [Batch, C, H, W] in range [0, 1]
            labels: Ground truth images [Batch, C, H, W] in range [0, 1]
        """
        outputs = outputs.clamp(0, 1).to(self.device)
        labels = labels.to(self.device)
        
        # Convert RGB to Y-channel (luminance) using ITU-R BT.601 standard
        # Y = 0.299*R + 0.587*G + 0.114*B
        if outputs.size(1) == 3:  # RGB image
            y_outputs = 0.299 * outputs[:, 0] + 0.587 * outputs[:, 1] + 0.114 * outputs[:, 2]
            y_labels = 0.299 * labels[:, 0] + 0.587 * labels[:, 1] + 0.114 * labels[:, 2]
        else:  # Grayscale image
            y_outputs = outputs[:, 0]
            y_labels = labels[:, 0]
        
        # Calculate MSE on Y-channel
        mse = torch.mean((y_outputs - y_labels) ** 2)
        
        # Accumulate
        self.total_mse += mse.item() * outputs.size(0)
        self.count += outputs.size(0)

    def compute(self):
        """
        Compute final PSNR normalized to [0.0, 1.0] range.
        
        Returns:
            float: Normalized PSNR in range [0.0, 1.0]
        """
        if self.count == 0:
            return 0.0
        
        avg_mse = self.total_mse / self.count
        
        # Perfect reconstruction
        if avg_mse == 0:
            return 1.0
        
        # Calculate PSNR in dB
        raw_psnr = 10 * math.log10(1.0 / avg_mse)
        
        # Normalize by 48 dB (professor's requirement)
        # This ensures metric is in [0.0, 1.0] range
        normalized_psnr = min(raw_psnr / 48.0, 1.0)
        
        return normalized_psnr
    
    def __call__(self, outputs, labels):
        """Convenience method for single-call evaluation"""
        self.update(outputs, labels)
        return self.compute()
    
    def result(self):
        """Get final result without updating"""
        return self.compute()


def create_metric(out_shape):
    """Factory function for metric creation"""
    return Net(out_shape)


# Alias for compatibility with training framework
Metric = Net
