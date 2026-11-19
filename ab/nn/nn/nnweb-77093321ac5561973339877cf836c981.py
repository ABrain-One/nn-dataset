import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class SPPBlock(nn.Module):
    """
    Spatial Pyramid Pooling Block.
    Replaces standard Global Average Pooling with a multi-scale pooling approach
    to capture global context while preserving some spatial structure.
    """
    def __init__(self, in_channels, out_channels, pool_sizes=(1, 2, 3)):
        super(SPPBlock, self).__init__()
        self.pool_sizes = pool_sizes
        # 1x1 conv to project the concatenated pooled features to out_channels
        total_pooled = in_channels * len(pool_sizes)
        self.conv = nn.Conv2d(total_pooled, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # x is (N, C, H, W)
        # We assume H=W for CIFAR-like inputs, but we use adaptive pooling to be safe.
        # For SPP, we typically pool to fixed sizes.
        # If H and W are different, AdaptiveAvgPool2d handles it.
        # However, standard SPP expects square inputs or handles rectangular by taking the min dimension?
        # PyTorch's AdaptiveAvgPool2d takes a tuple (H, W).
        # Let's just use the pool sizes as given.
        
        pooled = []
        for size in self.pool_sizes:
            # If size is larger than current spatial dim, it will just be the whole thing.
            # For CIFAR 32x32 -> 16 -> 8 -> 4.
            # At the end, spatial is 4x4.
            # Pooling to 3x3, 2x2, 1x1 is fine.
            # IMPORTANT: AdaptiveAvgPool2d requires a tuple (H, W).
            # If we pass an int, it treats it as (int, int) for square, but 
            # if the input is not square, it might behave unexpectedly or error if mixed types are used.
            # Actually, F.adaptive_avg_pool2d accepts int or tuple.
            # The error "Expected size 1 but got size 2" implies that one tensor became 1x1 and another 2x2?
            # No, AdaptiveAvgPool2d(1) makes it 1x1. AdaptiveAvgPool2d(2) makes it 2x2.
            # The error is in torch.cat.
            # Why would they have different sizes?
            # Ah, if the input spatial size is smaller than the pool size?
            # AdaptiveAvgPool2d does not upscale. If input is 1x1 and we ask for 2x2, it stays 1x1.
            # If input is 4x4, asking for 1, 2, 3 gives 1x1, 2x2, 3x3.
            # These have different spatial dimensions!
            # You cannot concatenate tensors of shape (N, C, 1, 1) and (N, C, 2, 2) along dim 1.
            # SPP usually projects each pool to a fixed size (e.g., 1x1) before concatenating,
            # OR it concatenates the spatial features which implies they must be the same size.
            # Standard SPP (e.g., in AlexNet) pools to different sizes (4x4, 2x2, 1x1) and then
            # the next layer is a fully connected layer that expects a fixed input size.
            # However, if we want to use a Conv2d after SPP, we need the spatial dimensions to match.
            # Common trick: Pool to different sizes, then Upsample/Resize to a common size, or
            # simply Pool to the SAME size (which is just GAP) - that defeats the purpose.
            
            # Alternative: Use the "SPP" logic where we pool to different sizes, but then we cannot
            # directly concatenate them if we want to keep them as images.
            # We can flatten the smaller ones? No, that breaks the conv.
            
            # Correct Implementation for Conv2d compatibility:
            # Pool to size s. If s > input_size, the output is input_size.
            # To make them compatible for concatenation, we must ensure they all have the same spatial size.
            # The easiest way is to pool everything to the MAX size, then downsample the larger ones?
            # No, standard SPP concatenates the pooled features.
            # If we want to feed this into a Conv2d, we should resize the pooled features to a common size,
            # e.g., the largest pool size, or just 1x1.
            
            # Let's resize all pooled features to the largest pool size in the list.
            max_size = max(self.pool_sizes)
            p = F.adaptive_avg_pool2d(x, (size, size))
            if p.shape[2] != max_size or p.shape[3] != max_size:
                # Upsample to max_size
                p = F.interpolate(p, size=(max_size, max_size), mode='bilinear', align_corners=False)
            pooled.append(p)
        
        # Concatenate along channel dimension
        # Now all tensors have shape (N, C, max_size, max_size)
        x_cat = torch.cat(pooled, dim=1)
        
        x = self.conv(x_cat)
        x = self.bn(x)
        x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    """
    Standard Residual Block with Depthwise Separable Convs.
    Structure:
    Shortcut: Identity (or 1x1 conv if dims change)
    Main: BN -> SiLU -> DWConv 3x3 -> BN -> SiLU -> PWConv 1x1
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        
        # Main Path
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        
        out = self.bn1(x)
        out = self.act1(out)
        out = self.dw_conv(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.pw_conv(out)
        out = self.bn3(out)
        
        return out + shortcut

class InvertedResidualBlock(nn.Module):
    """
    MobileNetV2-style block: Expand -> Depthwise -> Project
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4):
        super(InvertedResidualBlock, self).__init__()
        self.use_res = (stride == 1) and (in_channels == out_channels)
        
        hidden_dim = int(in_channels * expand_ratio)
        
        # Expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )
        
        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        input = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        if self.use_res:
            x = x + input
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        
        # Determine input channels
        if len(in_shape) == 4:
            c_in = in_shape[1]
            h, w = in_shape[2], in_shape[3]
        elif len(in_shape) == 3:
            if in_shape[0] == 3:
                c_in = in_shape[0]
                h, w = in_shape[1], in_shape[2]
            elif in_shape[2] == 3:
                c_in = in_shape[2]
                h, w = in_shape[0], in_shape[1]
            else:
                c_in = in_shape[0]
                h, w = in_shape[1], in_shape[2]
        else:
            c_in = 3
            h, w = 32, 32

        n_classes = out_shape[0] if out_shape else 10

        # Architecture Design:
        # Stem: Conv 3x3 -> BN -> SiLU
        # Stage 1: 2 InvertedResidualBlocks (channels: 32 -> 32), no downsample
        # Stage 2: 2 InvertedResidualBlocks (channels: 32 -> 64), downsample at start
        # Stage 3: 2 InvertedResidualBlocks (channels: 64 -> 128), downsample at start
        # Stage 4: 2 InvertedResidualBlocks (channels: 128 -> 256), downsample at start
        # Head: SPPBlock -> Global Avg Pool -> Linear
        
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32 channels, 2 blocks
        self.stage1 = nn.Sequential(
            InvertedResidualBlock(32, 32, stride=1),
            InvertedResidualBlock(32, 32, stride=1)
        )
        
        # Stage 2: 32 -> 64 channels, 2 blocks
        self.stage2 = nn.Sequential(
            InvertedResidualBlock(32, 64, stride=2),
            InvertedResidualBlock(64, 64, stride=1)
        )
        
        # Stage 3: 64 -> 128 channels, 2 blocks
        self.stage3 = nn.Sequential(
            InvertedResidualBlock(64, 128, stride=2),
            InvertedResidualBlock(128, 128, stride=1)
        )
        
        # Stage 4: 128 -> 256 channels, 2 blocks
        self.stage4 = nn.Sequential(
            InvertedResidualBlock(128, 256, stride=2),
            InvertedResidualBlock(256, 256, stride=1)
        )
        
        # SPP Head
        # Input to SPP: 256 channels, 4x4 spatial (assuming 32x32 input).
        # SPP pools to 1, 2, 3.
        # Fix: SPP output is max_size x max_size (3x3), so we don't need a separate GAP.
        # We flatten the 3x3 feature map.
        self.spp = SPPBlock(256, 256, pool_sizes=(1, 2, 3))
        
        # Classifier
        # Input to FC is 256 channels * 3 * 3 (since max pool size is 3)
        self.fc = nn.Linear(256 * 3 * 3, n_classes)
        
        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.spp(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        self.batch_size = prm.get('batch', 128)
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        try:
            total_steps = len(train_data)
        except (TypeError, AttributeError):
            total_steps = 1000
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=0.0
        )
        
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            
            self.optimizer.step()
            scheduler.step()