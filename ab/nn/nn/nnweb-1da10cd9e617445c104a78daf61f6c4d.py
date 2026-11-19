import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# --- MANDATORY FOR EVAL ENGINE ---
# These functions must be kept exactly as they are in the original model to improve.
# Since the "model to improve" isn't provided in the prompt text explicitly, I will reconstruct
# the standard "fixed" training setup described in the system prompt:
# "AdamW, a cosine learning-rate schedule within the one epoch, batch 128".
# I will implement a robust version that matches this description.

def supported_hyperparameters():
    return {'lr', 'momentum'}

class GatedMixingBlock(nn.Module):
    """
    A block that combines local convolutional mixing with global attention,
    modulated by a learned gating mechanism. This differs from standard
    ResNet or Fractal structures by using explicit gating and parallel
    attention/conv branches.
    """
    def __init__(self, channels: int, num_heads: int = 4, drop_prob: float = 0.0):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        
        # Branch 1: Local Convolution (Depthwise + Pointwise)
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # Branch 2: Global Attention (Simplified Linear Attention for efficiency)
        # Project to Q, K, V
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.attn_norm = nn.BatchNorm2d(channels)
        
        # Gating Mechanism: Determines how much of each branch to use
        # Input: Global average pooled features -> MLP -> Gate
        self.gate_mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, 2 * channels), # Output 2 channels worth for sigmoid gating
            nn.Sigmoid()
        )
        
        self.drop_path = nn.Dropout(drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 1. Local Branch
        local_out = self.local_conv(x)
        
        # 2. Global Attention Branch
        # Reshape for attention: [B, C, H*W]
        x_flat = x.flatten(2) # [B, C, HW]
        
        q = self.q_proj(x).flatten(2) # [B, C, HW]
        k = self.k_proj(x).flatten(2) # [B, C, HW]
        v = self.v_proj(x).flatten(2) # [B, C, HW]
        
        # Simple linear attention: softmax(QK^T/sqrt(d))V
        # To keep it cheap and stable, we use a simplified scaled dot product
        # scaled by sqrt(C)
        scale = 1.0 / (C ** 0.5)
        attn_weights = torch.bmm(q.transpose(1, 2), k) * scale # [B, HW, HW]
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to V
        attn_out = torch.bmm(attn_weights, v) # [B, HW, C]
        attn_out = attn_out.transpose(1, 2) # [B, C, HW]
        attn_out = attn_out.reshape(B, C, H, W)
        attn_out = self.attn_norm(attn_out)
        
        # 3. Gating
        # Compute global context for gating
        global_ctx = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1) # [B, C]
        gates = self.gate_mlp(global_ctx) # [B, 2C]
        
        gate_local = gates[:, :C].unsqueeze(-1).unsqueeze(-1) # [B, C, 1, 1]
        gate_global = gates[:, C:].unsqueeze(-1).unsqueeze(-1) # [B, C, 1, 1]
        
        # Combine branches
        out = gate_local * local_out + gate_global * attn_out
        
        # Residual connection with drop path
        out = self.drop_path(out) + x
        
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device

        c_in = 3 
        n_classes = out_shape[0] if out_shape else 10
        
        # Start with a stem that reduces resolution slightly and increases channels
        # CIFAR is 32x32. 
        # Stem: 3 -> 48, stride 1
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU()
        )
        
        # Architecture:
        # Stage 1: 48 channels, 32x32
        # Stage 2: 96 channels, 16x16 (via stride 2 conv)
        # Stage 3: 192 channels, 8x8 (via stride 2 conv)
        # Stage 4: 384 channels, 4x4 (via stride 2 conv)
        
        # We use GatedMixingBlocks.
        # To keep params < 5M, we need to be careful with attention size.
        # At 32x32, attention is 1024x1024. That's heavy.
        # We will use a "local attention" or reduce resolution before attention?
        # Actually, linear attention O(N^2) is bad for 32x32.
        # Let's modify GatedMixingBlock to use a cheaper attention or just global average pooling based gating without full matrix attention?
        # The prompt asks for strong building blocks. Attention is strong.
        # Let's use a "Linear Attention" approximation or restrict attention to a smaller spatial grid?
        # Or, we can just use the GatedMixingBlock but ensure it's efficient.
        # For CIFAR, 32x32 = 1024 tokens. 1024x1024 matrix is ~1M params per head if stored, but it's computed.
        # However, 4 heads * 1024 * 1024 * 4 bytes = 16MB memory per batch item? No, batch 128 -> 2GB. Too much.
        
        # Let's adjust GatedMixingBlock to use a more efficient attention mechanism:
        # "Linear Attention" using feature maps: exp(QK) is hard.
        # Let's use "Global Average Pooling" based context and "Local Conv" as the two branches, 
        # and replace the heavy matrix attention with a simpler "Global Mixing" via a 1x1 conv + Global Pooling + Broadcast?
        # That's basically a SE block. 
        # Let's stick to the GatedMixingBlock but make the attention part "Local Attention" (e.g. 4x4 windows) or just remove the matrix part and use a "Global Context" branch that is cheap.
        
        # Revised Plan for GatedMixingBlock:
        # Branch 1: Local Conv (3x3 DW + 1x1)
        # Branch 2: Global Context (AdaptiveAvgPool -> 1x1 Conv -> Upsample)
        # This is much cheaper and still captures global info.
        
        # Let's redefine GatedMixingBlock in the class below to be cheaper.
        
        self.stage1 = nn.Sequential(
            GatedMixingBlock(48, drop_prob=0.0),
            GatedMixingBlock(48, drop_prob=0.0)
        )
        
        # Transition 1: 48 -> 96, stride 2
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU()
        )
        
        self.stage2 = nn.Sequential(
            GatedMixingBlock(96, drop_prob=0.0),
            GatedMixingBlock(96, drop_prob=0.0)
        )
        
        # Transition 2: 96 -> 192, stride 2
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU()
        )
        
        self.stage3 = nn.Sequential(
            GatedMixingBlock(192, drop_prob=0.0),
            GatedMixingBlock(192, drop_prob=0.0)
        )
        
        # Transition 3: 192 -> 384, stride 2
        self.trans3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.GELU()
        )
        
        self.stage4 = nn.Sequential(
            GatedMixingBlock(384, drop_prob=0.0)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(384, n_classes)
        )
        
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.trans3(x)
        x = self.stage4(x)
        
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        # Store batch size
        self.batch_size = prm.get('batch', 128)
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        try:
            total_steps = len(train_data)
        except (TypeError, AttributeError):
            total_steps = 1000
        
        # Cosine schedule within 1 epoch
        # Start at lr, end at 0
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            
            self.optimizer.step()
            scheduler.step()

# Re-defining GatedMixingBlock to be the efficient version described in the thought process
# We must ensure the class used in Net is the correct one.
# The previous class definition had heavy attention. We replace it here.

class GatedMixingBlock(nn.Module):
    """
    Efficient Gated Mixing Block:
    - Branch 1: Local Convolution (DW + PW)
    - Branch 2: Global Context (AvgPool -> 1x1 Conv -> Upsample)
    - Gating: Learned per-channel gates from global context
    """
    def __init__(self, channels: int, drop_prob: float = 0.0):
        super().__init__()
        self.channels = channels
        self.drop_prob = drop_prob
        
        # Branch 1: Local
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # Branch 2: Global Context
        # Pool to 1x1, project, then upsample back to original size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # Gating Mechanism
        # Input: Global Avg Pool of X -> MLP -> 2 * Channels
        self.gate_mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, 2 * channels),
            nn.Sigmoid()
        )
        
        self.drop_path = nn.Dropout(drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Local Branch
        local_out = self.local_conv(x)
        
        # 2. Global Branch
        # Pool
        g = self.global_pool(x) # [B, C, 1, 1]
        g = self.global_proj(g) # [B, C, 1, 1]
        # Upsample to [B, C, H, W]
        global_out = F.interpolate(g, size=(H, W), mode='bilinear', align_corners=False)
        
        # 3. Gating
        # Context for gating is the pooled features
        global_ctx = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1) # [B, C]
        gates = self.gate_mlp(global_ctx) # [B, 2C]
        
        gate_local = gates[:, :C].unsqueeze(-1).unsqueeze(-1) # [B, C, 1, 1]
        gate_global = gates[:, C:].unsqueeze(-1).unsqueeze(-1) # [B, C, 1, 1]
        
        # Combine
        out = gate_local * local_out + gate_global * global_out
        
        # Residual
        out = self.drop_path(out) + x
        
        return out