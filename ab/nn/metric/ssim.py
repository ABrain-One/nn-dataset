
import torch
import torch.nn.functional as F

class Net:
    """
    SSIM metric for Super Resolution.
    - Computes SSIM on Y-channel (luminance) following standard SR evaluation
    - SSIM is naturally in [0.0, 1.0] range (1.0 = perfect similarity)
    - Implements simplified SSIM calculation
    """
    def __init__(self, out_shape=None):
        self.reset()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # SSIM constants
        self.C1 = (0.01) ** 2  # Stability constant for luminance
        self.C2 = (0.03) ** 2  # Stability constant for contrast

    def reset(self):
        """Clear accumulated metrics before new evaluation"""
        self.total_ssim = 0.0
        self.count = 0

    def _ssim_single_channel(self, img1, img2):
        """
        Calculate SSIM for single channel images.
        
        Args:
            img1, img2: Tensors of shape [B, H, W]
            
        Returns:
            SSIM value in range [0, 1]
        """
        # Add channel dimension for conv2d: [B, H, W] -> [B, 1, H, W]
        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)
        
        # Create Gaussian kernel for local statistics
        kernel_size = 11
        sigma = 1.5
        kernel = self._gaussian_kernel(kernel_size, sigma).to(img1.device)
        
        # Calculate local means
        mu1 = F.conv2d(img1, kernel, padding=kernel_size//2, groups=1)
        mu2 = F.conv2d(img2, kernel, padding=kernel_size//2, groups=1)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Calculate local variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, kernel, padding=kernel_size//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, kernel, padding=kernel_size//2, groups=1) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel, padding=kernel_size//2, groups=1) - mu1_mu2
        
        # SSIM formula
        numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        
        ssim_map = numerator / denominator
        
        # Return mean SSIM
        return ssim_map.mean()
    
    def _gaussian_kernel(self, kernel_size, sigma):
        """Create 2D Gaussian kernel"""
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]

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
        
        # Calculate SSIM on Y-channel
        ssim_value = self._ssim_single_channel(y_outputs, y_labels)
        
        # Accumulate
        self.total_ssim += ssim_value.item() * outputs.size(0)
        self.count += outputs.size(0)

    def compute(self):
        """
        Compute final SSIM.
        
        Returns:
            float: SSIM in range [0.0, 1.0] (1.0 = perfect similarity)
        """
        if self.count == 0:
            return 0.0
        
        avg_ssim = self.total_ssim / self.count
        
        # SSIM is already in [0, 1] range, no normalization needed
        return max(0.0, min(1.0, avg_ssim))
        
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
