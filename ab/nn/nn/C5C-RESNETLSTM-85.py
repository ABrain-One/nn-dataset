import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


def supported_hyperparameters():
    return {'lr','momentum'}



class BaseBlock(nn.Module):
    """Basic building block for neural networks."""
    def __init__(self, in_channels: int, out_channels: int, stride: int=1,
                 activation_layer: Optional[nn.Module]=None,
                 norm_layer: Optional[nn.Module]=None) -> None:
        super().__init__()
        self.activation = activation_layer or nn.ReLU(inplace=True)
        self.norm = norm_layer or nn.BatchNorm2d
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                             stride=stride, padding=1, bias=False)
        self.bn = self.norm(out_channels)
        
        self.initialize_weights()
    
    def initialize_weights(self) -> None:
        """Initialize weights using Kaiming normalization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class BottleNeck(BaseBlock):
    """Improved residual block variant."""
    def __init__(self, in_channels: int, out_channels: int, 
                 expansion_ratio: float=4, groups: int=1,
                 base_width: int=64, activation_layer: Optional[nn.Module]=None,
                 norm_layer: Optional[nn.Module]=None) -> None:
        super().__init__(in_channels, out_channels, norm_layer=norm_layer)
        
        self.expansion_ratio = expansion_ratio
        self.groups = groups
        self.base_width = base_width
        
        mid_channels = int(out_channels * expansion_ratio)
        
        self.shortcut = nn.Sequential() if in_channels == mid_channels else \
                         nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        
        self.shortcut_bn = norm_layer(mid_channels) if in_channels != mid_channels else None
        
        _layer_list = [
            BaseBlock(in_channels, mid_channels // expansion_ratio, 
                      kernel_size=1, stride=1),
            BaseBlock(mid_channels // expansion_ratio, mid_channels // expansion_ratio, 
                      kernel_size=3, stride=stride, groups=self.groups),
            BaseBlock(mid_channels // expansion_ratio, out_channels, kernel_size=1)
        ]
        
        self.layer = nn.Sequential(*_layer_list)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        
        if self.shortcut_bn is not None:
            shortcut = self.shortcut_bn(shortcut)
            
        y = self.layer(x)
        return shortcut + y


class SpatialAttentionLSTM(nn.Module):
    """LSTM model with spatial attention capabilities."""
    def __init__(self, in_channels: int, out_channels: int, 
                 hidden_size: int=768, layers: int=1, dropout: float=0.2,
                 activation_layer: Optional[nn.Module]=None) -> None:
        super().__init__()
        
        self.activation = activation_layer or nn.ReLU(inplace=True)
        
        self.conv_att = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                 padding=1, bias=True)
        self.bn_att = nn.BatchNorm2d(out_channels)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.rnn_cell = nn.LSTMCell(out_channels, hidden_size, num_layers=layers,
                                   bias=True, batch_first=True)
        
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        
        self.initialize_weights()
    
    def initialize_weights(self) -> None:
        """Initialize weights properly with specific norms."""
        for name, module in self.named_children:
            if 'bn' in name or 'conv_att' in name:
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
            else:  # LSTM cell weights
                if isinstance(module, nn.LSTMCell):
                    for i, (weight, _) in enumerate([(attr, getattr(module, attr)) 
                                                     for attr in ['weight_ih', 'weight_hh']]):
                        if weight.dim() == 4:
                            weight = weight.squeeze()
                        if weight.size(0) == self.hidden_size and weight.size(1) == 4 * self.hidden_size:
                            fan_in = self.hidden_size * 4
                            fan_out = self.hidden_size
                            u = nn.init.orthogonal_(torch.Tensor(fan_in, fan_out), gain=1)
                            weight.data.copy_(u.data[:weight.size(0), :weight.size(1)])
                    
                    for _, bias in [('bias_ih', getattr(module, 'bias_ih')), ('bias_hh', getattr(module, 'bias_hh'))]:
                        if bias.dim() == 4:
                            bias = bias.squeeze()
                        fan_in = self.hidden_size * 4
                        bias = bias.new_zeros(*bias.size())
                        u = (1/(fan_in**0.5)) * torch.ones(bias.size())
                        if bias.dim() == 2:
                            u = u[:, self.hidden_size//2:].clone()
                        bias.data.copy_(u.data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_att = self.conv_att(x)
        x_att = self.bn_att(x_att)
        x_att = self.activation(x_att)
        x_att = self.global_pool(x_att).squeeze()
        
        h0 = torch.zeros(self.rnn_cell.num_layers, x_att.size(0), self.hidden_size).to(x_att.device)
        c0 = torch.zeros(self.rnn_cell.num_layers, x_att.size(0), self.hidden_size).to(x_att.device)
        
        output, _ = self.rnn_cell(x_att.unsqueeze(0), (h0, c0))
        output = output.squeeze()
        
        return output


class Net(nn.Module):
    """Main network class for captioning."""
    def __init__(self, in_shape: Tuple[int, int, int], out_shape: Tuple[int, int, int], 
                 prm: Dict[str, float], device: str) -> None:
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device
        
        # Backbone network
        self.backbone = nn.Sequential(
            BottleNeck(3, 64, expansion_ratio=4),
            BottleNeck(64, 128, expansion_ratio=4),
            BottleNeck(128, 256, expansion_ratio=4),
            BottleNeck(256, 512, expansion_ratio=4)
        )
        
        # Attention LSTM
        self.attention = SpatialAttentionLSTM(512, 768, hidden_size=512, layers=2, dropout=0.5)
        
        # Fully connected layer for output
        self.fc = nn.Linear(768, out_shape[0] * out_shape[0] * out_shape[2])
        
        self.to(device)
    
    def train_setup(self, prm: Dict[str, float]) -> None:
        """Set up the model for training."""
        self.prm = prm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=prm['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    def learn(self, train_data: torch.Tensor) -> None:
        """Train the model on the provided data."""
        self.train_setup(self.prm)
        train_data = train_data.to(self.device)
        
        for epoch in range(10):  # Example training loop
            self.backbone.train()
            self.attention.train()
            self.fc.train()
            
            outputs = self(train_data)
            loss = F.mse_loss(outputs, torch.zeros_like(outputs))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.scheduler.step(loss)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"
        assert x.dim() == 4, "Input must be 4D tensor"
        
        # Backbone processing
        x = self.backbone(x)
        
        # Attention LSTM processing
        x = self.attention(x)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x