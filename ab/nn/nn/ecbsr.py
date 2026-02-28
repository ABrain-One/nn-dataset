"""
ECBSR: Edge-oriented Convolution Block for Real-time Super Resolution
Paper: "Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices"
Source: ACM MM 2021
GitHub: https://github.com/xindongzhang/ECBSR

Key Features:
- Re-parameterizable Edge-oriented Convolution Block (ECB)
- Multi-path feature extraction during training
- Single 3×3 convolution during inference (zero cost)
- Real-time performance on mobile SOCs (Snapdragon 865, Dimensity 1000+)
- Parameters: ~600K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    """Return supported hyperparameters for Optuna optimization"""
    return {'lr'}


class SeqConv3x3(nn.Module):
    """Sequential Convolution for re-parameterization"""
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super().__init__()
        
        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        
        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            
            conv1 = nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
            
        elif self.type in ['conv1x1-sobelx', 'conv1x1-sobely', 'conv1x1-laplacian']:
            conv0 = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            
            # Init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            self.bias = nn.Parameter(bias)
            
            # Init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                if self.type == 'conv1x1-sobelx':
                    self.mask[i, 0, 0, 0] = 1.0
                    self.mask[i, 0, 1, 0] = 2.0
                    self.mask[i, 0, 2, 0] = 1.0
                    self.mask[i, 0, 0, 2] = -1.0
                    self.mask[i, 0, 1, 2] = -2.0
                    self.mask[i, 0, 2, 2] = -1.0
                elif self.type == 'conv1x1-sobely':
                    self.mask[i, 0, 0, 0] = 1.0
                    self.mask[i, 0, 0, 1] = 2.0
                    self.mask[i, 0, 0, 2] = 1.0
                    self.mask[i, 0, 2, 0] = -1.0
                    self.mask[i, 0, 2, 1] = -2.0
                    self.mask[i, 0, 2, 2] = -1.0
                elif self.type == 'conv1x1-laplacian':
                    self.mask[i, 0, 0, 1] = 1.0
                    self.mask[i, 0, 1, 0] = 1.0
                    self.mask[i, 0, 1, 2] = 1.0
                    self.mask[i, 0, 2, 1] = 1.0
                    self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
    
    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            # Conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # Explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # Conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # Explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # Conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1
    
    def rep_params(self):
        """Get re-parameterized kernel and bias"""
        device = self.k0.get_device()
        if device < 0:
            device = None
        
        if self.type == 'conv1x1-conv3x3':
            # Re-param conv kernel
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # Re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1,) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # Re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # Re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1,) + b1
        return RK, RB


class ECB(nn.Module):
    """Edge-oriented Convolution Block"""
    def __init__(self, inp_planes, out_planes, depth_multiplier, act_type='prelu', with_idt=False):
        super().__init__()
        
        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        
        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False
        
        self.conv3x3 = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.inp_planes, self.out_planes, -1)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.inp_planes, self.out_planes, -1)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes, -1)
        
        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation is not supported!')
    
    def forward(self, x):
        if self.training:
            y = (self.conv3x3(x) + 
                 self.conv1x1_3x3(x) + 
                 self.conv1x1_sbx(x) + 
                 self.conv1x1_sby(x) + 
                 self.conv1x1_lpl(x))
            if self.with_idt:
                y += x
        else:
            RK, RB = self.rep_params()
            y = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
        
        if self.act_type != 'linear':
            y = self.act(y)
        return y
    
    def rep_params(self):
        """Get re-parameterized kernel and bias"""
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        RK, RB = (K0+K1+K2+K3+K4), (B0+B1+B2+B3+B4)
        
        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB


class ECBSR(nn.Module):
    """ECBSR Architecture"""
    def __init__(self, module_nums=4, channel_nums=16, with_idt=True, act_type='prelu', scale=4, colors=3):
        super().__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.with_idt = with_idt
        self.act_type = act_type
        
        backbone = []
        backbone += [ECB(self.colors, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt=self.with_idt)]
        for i in range(self.module_nums):
            backbone += [ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt=self.with_idt)]
        backbone += [ECB(self.channel_nums, self.colors*self.scale*self.scale, depth_multiplier=2.0, act_type='linear', with_idt=self.with_idt)]
        
        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        y = self.backbone(x)
        y = self.upsampler(y)
        return y


class Net(nn.Module):
    """Wrapper for LEMUR/NN Dataset framework"""
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        
        # Extract parameters
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        
        # Get channels from shape
        if len(in_shape) == 4:
            in_channels = in_shape[1]
        else:
            in_channels = 3
            
        if len(out_shape) == 4:
            out_channels = out_shape[1]
        else:
            out_channels = 3
        
        # Calculate upscale factor
        if len(in_shape) >= 3 and len(out_shape) >= 3:
            upscale = out_shape[-1] // in_shape[-1]
        else:
            upscale = 4
        
        # Model configuration (optimized for mobile)
        module_nums = 4  # Number of ECB blocks
        channel_nums = 16  # Feature channels
        
        # Create model
        self.model = ECBSR(
            module_nums=module_nums,
            channel_nums=channel_nums,
            with_idt=True,
            act_type='prelu',
            scale=upscale,
            colors=in_channels
        )
        
        # Loss and optimizer
        self.criterion = nn.L1Loss()
        self.lr = prm.get('lr', 1e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
    def train_setup(self, prm):
        """Setup for training"""
        self.model.train()
        
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def learn(self, data_roll):
        """Training step"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for lr_img, hr_img in data_roll:
            lr_img = lr_img.to(self.device)
            hr_img = hr_img.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            sr_img = self.model(lr_img)
            
            # Compute loss
            loss = self.criterion(sr_img, hr_img)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
