import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {'lr','momentum'}




class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        
        # Parse input/output shapes
        channel_in = int(in_shape[1])
        height, width = in_shape[2:]
        
        # Parse hyperparameters
        self.vocab_size = out_shape[0][0][0]  # Assuming out_shape is [vocab_size]
        self.hidden_size = int(prm.get('hidden_size', 768))  # Minimum 640 according to constraint
        self.num_heads = int(prm.get('num_heads', 8))        # Must divide hidden_size
        
        # Validate hyperparameter choices
        assert self.hidden_size >= 640, "Hidden size must be at least 640"
        assert self.num_heads > 0 and self.num_heads <= 16, "Number of heads must be positive and not exceed 16"
        
        # Select encoder block type
        self.encoder_block_type = str(prm.get('encoder_block_type', 'bagnet')).lower()
        
        # Initialize components based on encoder block type
        self._setup_encoder_components()
        
        # Setup dropout probability if provided
        self.dropout_prob = float(prm.get('dropout', 0.1))
        
        # Final classification layer (no softmax here, keep it for later)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, self.hidden_size),  # Small linear layer to match the adapter requirement
            nn.Dropout(p=self.dropout_prob if 'dropout' in prm and prm['dropout'] > 0 else 0.0)
        )

    def _setup_encoder_components(self):
        """Set up the appropriate encoder backbone"""
        if self.encoder_block_type == 'bagnet':
            self.encoder = self.build_bag_net_encoder()
        else:
            self.encoder = self.build_simple_encoder()
            
    def build_bag_net_encoder(self):
        """Build BagNet-style encoder with bottleneck units"""
        return nn.Sequential(
            # Initial convolution layer
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Stage 1
            BagNetStage(64, 64, 3, 1),
            
            # Stage 2
            BagNetStage(64, 128, 3, 2),
            
            # Stage 3
            BagNetStage(128, 256, 3, 2),
            
            # Stage 4
            BagNetStage(256, 512, 3, 2)
        )

    def build_simple_encoder(self):
        """Build a Simpler alternative encoder"""
        return nn.Sequential(
            # Basic residual encoder without special blocks
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Additional layers depending on chosen hidden_size
            *(self._build_residual_blocks(self.hidden_size))
        )
    
    def _build_residual_blocks(self, target_size):
        """Build multiple residual blocks depending on hidden_size"""
        return [
            # Increasing capacity as hidden_size grows
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1,1))
        ]
        
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999)
        )

            
    def learn(self, train_data):
        """Train the model on the provided data"""
        # Convert train_data to DataLoader if needed
        if not hasattr(train_data, '__iter__'):
            raise ValueError("train_data must be iterable")
            
        # Training loop
        self.train()
        for inputs, targets in train_data:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self(inputs)
            
            # Compute loss (assuming classification with cross entropy)
            loss = F.cross_entropy(outputs, targets)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def forward(self, x):
        # Teacher forcing: directly use the input for the decoder
        # Forward pass through encoder
        x = self.encoder(x)
        
        # Forward pass through classifier
        x = self.classifier(x)
        
        return x


class BagNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=True):
        super(BagNetUnit, self).__init__()
        self.activation = activation
        self.main_path = BagNetBottleneck(in_channels, out_channels, kernel_size, stride)
        self.shortcut_path = None
        
        if activation and in_channels != out_channels and stride == 1:
            self.shortcut_path = self.conv1x1_block(in_channels, out_channels)
        elif activation and (in_channels != out_channels or stride != 1):
            self.shortcut_path = self.resize_identity(in_channels, out_channels, stride)

    def resize_identity(self, in_channels, out_channels, stride):
        """Resize the shortcut connection appropriately"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def conv1x1_block(self, in_channels, out_channels, activation=True):
        """1x1 convolution helper with optional activation"""
        _layer_list = [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            _layer_list.append(nn.ReLU(inplace=True))
        return nn.Sequential(*_layer_list)

    def forward(self, x):
        identity = x
        
        if self.shortcut_path:
            if isinstance(self.shortcut_path, tuple) or isinstance(self.shortcut_path, list):
                # Handle resizing shortcut
                identity = self.shortcut_path[0](identity)
                if len(self.shortcut_path) == 2:
                    identity = self.shortcut_path[1](identity)
            else:
                identity = self.shortcut_path(x)
                
        x = self.main_path(x)
        
        if self.activation:
            x = F.relu(x)
            
        return x + identity


class BagNetStage(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(BagNetStage, self).__init__()
        self.add_module("unit1", BagNetUnit(in_channels, out_channels, kernel_size, stride, activation=True))


class DecoderSEBlock(nn.Module):
    """Squeeze-and-Excitation block for attention gates"""
    def __init__(self, input_channels, squeeze_channels=None, activation=nn.ReLU):
        super(DecoderSEBlock, self).__init__()
        if squeeze_channels is None:
            squeeze_channels = input_channels // 4
            
        self.squeeze = nn.Sequential(
            nn.Conv2d(input_channels, squeeze_channels, kernel_size=1),
            nn.BatchNorm2d(squeeze_channels),
            activation(inplace=True)
        )
        
        self.excite = nn.Sequential(
            nn.Conv2d(squeeze_channels, input_channels, kernel_size=1),
            nn.BatchNorm2d(input_channels)
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, output_size=1)
        y = self.relu(self.squeeze(y))
        y = self.excite(y)
        return x * (1 + y)


class DecoderCBAM(nn.Module):
    """Convolutional Block Attention Module for dynamic filtering"""
    def __init__(self, channels, reduction_ratio=4, kernel_size=3, pool='avg'):
        super(DecoderCBAM, self).__init__()
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.BatchNorm2d(1)
        )
        
        self.pool = pool

    def forward(self, x):
        in_size = x.size()
        y = F.adaptive_avg_pool2d(x, output_size=1) if self.pool == 'avg' else \
            F.adaptive_max_pool2d(x, output_size=1)
        y = y.permute(0, 2, 3, 1).contiguous().reshape(-1, self.channels)
        y = self.mlp(y)
        y = y.reshape(in_size[0], -1, 1, 1)
        y = self.spatial_attention(y)
        y = y.expand(-1, self.channels, -1, -1)
        return x * y


class BagNetBottleneck(nn.Module):
    """Bottleneck block for BagNet"""
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(BagNetBottleneck, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut path
        self.shortcut_path = None
        
        if in_channels != out_channels or stride != 1:
            self.shortcut_path = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = x
        
        x = self.main_path(x)
        
        if self.shortcut_path:
            identity = self.shortcut_path(x)
            
        return x + identity


def main():
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(in_shape=(3, 224, 224), out_shape=[1000], prm={'lr': 0.001, 'momentum': 0.9}, device=device)
    model.train_setup({'lr': 0.001, 'momentum': 0.9})
    
    # Dummy data for demonstration
    train_data = [(torch.randn(1, 3, 224, 224), torch.randint(0, 1000, (1,))) for _ in range(10)]
    model.learn(train_data)
    
    # Test forward pass
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()