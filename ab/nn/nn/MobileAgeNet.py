"""
MobileAgeNet - A Lightweight Neural Network for Facial Age Estimation

This model is designed for real-time mobile deployment while maintaining
high accuracy for age estimation from facial images.

Architecture Features:
- MobileNetV3-inspired inverted residual blocks with squeeze-and-excitation
- Depthwise separable convolutions for efficiency
- Hard-swish activation for better accuracy with low computational cost
- Global average pooling to reduce parameters
- Regression head for continuous age prediction

Target Specifications:
- Input: 224x224 RGB face images
- Output: Single age value (regression)
- Parameters: ~1.5M (suitable for mobile)
- Optimized for TFLite/ONNX export
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


def supported_hyperparameters():
    """Return the set of hyperparameters this model supports."""
    return {'lr', 'momentum', 'dropout'}


class HardSwish(nn.Module):
    """Hard-Swish activation function - efficient approximation of Swish."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.0) / 6.0


class HardSigmoid(nn.Module):
    """Hard-Sigmoid activation function for SE blocks."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0) / 6.0


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Lightweight version optimized for mobile deployment.
    """
    
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)
        self.activation = HardSigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale))
        scale = self.activation(self.fc2(scale))
        return x * scale


class InvertedResidual(nn.Module):
    """
    MobileNetV3-style inverted residual block with optional SE.
    Uses depthwise separable convolutions for efficiency.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expansion_factor: float = 4,
        use_se: bool = True
    ):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_channels = int(in_channels * expansion_factor)
        padding = (kernel_size - 1) // 2
        
        layers = []
        
        # Expansion phase (pointwise conv)
        if expansion_factor != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                HardSwish()
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(
                hidden_channels, hidden_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                groups=hidden_channels, bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            HardSwish()
        ])
        
        # Squeeze-and-Excitation
        if use_se:
            layers.append(SqueezeExcitation(hidden_channels))
        
        # Projection phase (pointwise conv)
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class Net(nn.Module):
    """
    MobileAgeNet: Lightweight CNN for facial age estimation.
    
    Designed for:
    - High accuracy on IMDB-WIKI dataset
    - Real-time inference on mobile devices
    - Easy export to TFLite/ONNX
    
    Architecture:
    - Initial convolution: 3 -> 16 channels
    - 4 stages of inverted residuals with increasing channels
    - Global average pooling
    - Fully connected regression head
    """
    
    def train_setup(self, prm: Dict[str, Any]) -> None:
        """Setup training configuration: loss function toward optimizer."""
        self.to(self.device)
        # L1 Loss (MAE) is more robust to outliers in age labels
        self.criteria = (nn.L1Loss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4
        )
    
    def learn(self, train_data: Any) -> None:
        """Perform one epoch of training."""
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float()
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            self.optimizer.step()
    
    def __init__(
        self,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        prm: Dict[str, Any],
        device: torch.device
    ) -> None:
        """
        Initialize MobileAgeNet.
        
        Args:
            in_shape: Input tensor shape (batch, channels, height, width)
            out_shape: Output shape, typically (1,) for age regression
            prm: Hyperparameters dict with 'lr', 'momentum', 'dropout'
            device: Torch device (cuda/cpu)
        """
        super().__init__()
        self.device = device
        dropout = prm.get('dropout', 0.2)
        
        # Extract number of channels from in_shape
        # in_shape is (batch, C, H, W) format from get_in_shape()
        in_channels = in_shape[1] if len(in_shape) >= 2 else 3
        
        # Initial convolution - extract low-level features
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish()
        )
        
        # Stage 1: 16 -> 24 channels, stride 2
        self.stage1 = nn.Sequential(
            InvertedResidual(16, 24, kernel_size=3, stride=2, expansion_factor=4, use_se=True),
            InvertedResidual(24, 24, kernel_size=3, stride=1, expansion_factor=3, use_se=True),
        )
        
        # Stage 2: 24 -> 40 channels, stride 2
        self.stage2 = nn.Sequential(
            InvertedResidual(24, 40, kernel_size=5, stride=2, expansion_factor=3, use_se=True),
            InvertedResidual(40, 40, kernel_size=5, stride=1, expansion_factor=3, use_se=True),
            InvertedResidual(40, 40, kernel_size=5, stride=1, expansion_factor=3, use_se=True),
        )
        
        # Stage 3: 40 -> 80 channels, stride 2
        self.stage3 = nn.Sequential(
            InvertedResidual(40, 80, kernel_size=3, stride=2, expansion_factor=6, use_se=False),
            InvertedResidual(80, 80, kernel_size=3, stride=1, expansion_factor=2.5, use_se=False),
            InvertedResidual(80, 80, kernel_size=3, stride=1, expansion_factor=2.3, use_se=False),
            InvertedResidual(80, 112, kernel_size=3, stride=1, expansion_factor=6, use_se=True),
        )
        
        # Stage 4: 112 -> 160 channels, stride 2  
        self.stage4 = nn.Sequential(
            InvertedResidual(112, 160, kernel_size=5, stride=2, expansion_factor=6, use_se=True),
            InvertedResidual(160, 160, kernel_size=5, stride=1, expansion_factor=6, use_se=True),
        )
        
        # Final convolution to increase features before pooling
        self.final_conv = nn.Sequential(
            nn.Conv2d(160, 960, kernel_size=1, bias=False),
            nn.BatchNorm2d(960),
            HardSwish()
        )
        
        # Age regression head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(960, 256),
            HardSwish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, out_shape[0])  # Output: predicted age
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using best practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 3, H, W)
            
        Returns:
            Predicted age tensor of shape (batch, 1)
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.final_conv(x)
        x = self.head(x)
        return x
    
    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
