"""
Mean Absolute Error (MAE) Metric for Age Regression

This metric measures the average absolute difference between predicted ages
and ground truth ages. Lower MAE indicates better performance.

The metric is normalized to return a value between 0 and 1, where:
- 1.0 = Perfect prediction (MAE = 0)
- 0.0 = Very poor prediction (MAE >= max_mae_threshold)

For age estimation, a MAE of 5 years or less is generally considered good,
and MAE of 3 years or less is considered excellent.
"""
import torch


class Net:
    """
    Mean Absolute Error metric for age regression tasks.
    
    Computes MAE between predicted and actual ages, then normalizes
    to a 0-1 accuracy score where higher is better.
    """
    
    def __init__(self, max_mae_threshold: float = 20.0):
        """
        Initialize the MAE metric.
        
        Args:
            max_mae_threshold: MAE value at which accuracy becomes 0.
                              Default is 20 years for age estimation.
        """
        self.name = "mae"
        self.max_mae_threshold = max_mae_threshold
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics for a new evaluation epoch."""
        self._total_abs_error = 0.0
        self._total_samples = 0
    
    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Update the metric with a batch of predictions and labels.
        
        Args:
            outputs: Predicted ages, shape (batch_size, 1) or (batch_size,)
            labels: Ground truth ages, shape (batch_size, 1) or (batch_size,)
        """
        # Flatten tensors to 1D
        outputs = outputs.view(-1)
        labels = labels.view(-1)
        
        # Compute absolute errors
        abs_errors = torch.abs(outputs - labels)
        
        # Accumulate
        self._total_abs_error += abs_errors.sum().item()
        self._total_samples += outputs.size(0)
    
    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Make the metric callable for compatibility with the training framework.
        
        Args:
            outputs: Predicted ages
            labels: Ground truth ages
        
        Returns:
            Tuple of (total_abs_error, total_samples) for compatibility
        """
        self.update(outputs, labels)
        return self._total_abs_error, self._total_samples
    
    def result(self) -> float:
        """
        Get the final normalized accuracy score (alias for compute).
        
        Returns:
            A float between 0 and 1, where 1 indicates perfect prediction.
        """
        return self.compute()
    
    def compute(self) -> float:
        """
        Compute the final normalized accuracy score.

        Returns:
            A float between 0 and 1, where 1 indicates perfect prediction
            and 0 indicates MAE >= max_mae_threshold.
        """
        if self._total_samples == 0:
            return 0.0

        mae_years = self._total_abs_error / self._total_samples
        accuracy = max(0.0, 1.0 - (mae_years / self.max_mae_threshold))

        return accuracy
    
    def get_mae(self) -> float:
        """
        Get the raw MAE value (not normalized).
        
        Returns:
            Mean Absolute Error in years.
        """
        if self._total_samples == 0:
            return float('inf')
        return self._total_abs_error / self._total_samples


def create_metric(out_shape=None):
    """
    Factory function to create an MAE metric instance.
    
    This function is required by the nn-dataset framework for dynamic metric loading.
    
    Args:
        out_shape: Output shape (not used for MAE, kept for compatibility)
    
    Returns:
        An instance of the MAE metric.
    """
    return Net(max_mae_threshold=20.0)
