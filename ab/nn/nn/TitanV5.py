import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ab.nn.util.Classes import DataRoll
from ab.nn.nn.HSRv4 import HSR_Godzilla

# Empty set disables Optuna's random LR overriding
def supported_hyperparameters():
    return set()

class ECBSR_RepBlock(nn.Module):
    """
    [ECBSR Logic] Multi-branch edge-oriented reparameterizable block.
    """
    def __init__(self, in_channels, out_channels, groups=1):
        super(ECBSR_RepBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.deploy = False
        
        # Training Branches
        self.rbr_dense = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups)
        
        # Deploy Branch
        self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        self.rbr_reparam.requires_grad_(False)

    def forward(self, x):
        if self.deploy:
            return self.rbr_reparam(x)
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.in_channels == self.out_channels:
            out = out + x
        return out

    def reparameterize(self):
        if self.deploy:
            return
        kernel3x3, bias3x3 = self.rbr_dense.weight, self.rbr_dense.bias
        kernel1x1, bias1x1 = self.rbr_1x1.weight, self.rbr_1x1.bias
        kernel1x1_padded = F.pad(kernel1x1, [1, 1, 1, 1])
        kernel_id = 0
        if self.in_channels == self.out_channels:
            kernel_id = torch.zeros_like(kernel3x3)
            in_c_per_group = self.in_channels // self.groups
            for i in range(self.out_channels):
                kernel_id[i, i % in_c_per_group, 1, 1] = 1.0
        self.rbr_reparam.weight.data = kernel3x3 + kernel1x1_padded + kernel_id
        self.rbr_reparam.bias.data = (bias3x3 if bias3x3 is not None else 0) + (bias1x1 if bias1x1 is not None else 0)
        self.deploy = True

class Splitless_DistillationBlock(nn.Module):
    """
    [Split-less Sequential Logic]
    """
    def __init__(self, channels, groups=4):
        super().__init__()
        self.c1 = ECBSR_RepBlock(channels, channels, groups=groups)
        self.c2 = ECBSR_RepBlock(channels, channels, groups=groups)
        self.c3 = ECBSR_RepBlock(channels, channels, groups=groups)
        self.c4 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups)
        self.fuse_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.c1(x))
        out = self.act(self.c2(out))
        out = self.act(self.c3(out))
        out = self.c4(out)
        out = self.fuse_conv(out)
        return out + x

class TitanV7_Lite_Custom(nn.Module):
    """
    TitanV7_Lite_Custom: f=32, n_blocks=3 for <100ms 4K NPU target.
    Uses PixelShuffle (maps to DEPTH_TO_SPACE on NPU).
    """
    def __init__(self, f=32, n_blocks=3, upscale=3, groups=4): 
        super(TitanV7_Lite_Custom, self).__init__()
        self.upscale = upscale
        self.deploy_mode = False 
        
        self.conv_input = nn.Conv2d(3, f, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[Splitless_DistillationBlock(f, groups=groups) for _ in range(n_blocks)])
        self.conv_mid = nn.Conv2d(f, f, kernel_size=3, padding=1)
        
        # PixelShuffle upsampler -> maps to DEPTH_TO_SPACE on Snapdragon NPU
        self.upsampler = nn.Sequential(
            nn.Conv2d(f, 3 * (upscale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale)
        )
        
        # Zero-Init: start from bilinear baseline (24.6 dB)
        nn.init.zeros_(self.upsampler[0].weight)
        nn.init.zeros_(self.upsampler[0].bias)

    def forward(self, x):
        h = self.conv_input(x)
        out = self.body(h)
        out = self.conv_mid(out) + h
        out = self.upsampler(out)
        if self.deploy_mode:
            base = F.interpolate(x, size=(2160, 3840), mode='bilinear', align_corners=False)
        else:
            base = F.interpolate(x, scale_factor=float(self.upscale), mode='bilinear', align_corners=False)
        return torch.clamp(out + base, 0.0, 1.0)

    def deploy(self):
        self.deploy_mode = True  
        for m in self.modules():
            if hasattr(m, 'reparameterize'):
                m.reparameterize()

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.scale = 3
        self.current_epoch = 0
        
        # 1. Student Model (f=32, n_blocks=3, NPU-optimized)
        self.model = TitanV7_Lite_Custom(f=32, n_blocks=3, upscale=self.scale, groups=4).to(device)
        
        # 2. Godzilla Teacher (frozen, f=180)
        self.teacher = HSR_Godzilla(feature_channels=180, n_groups=5, n_blocks=8, upscale=self.scale).to(device)
        self.teacher.eval() 
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # 3. Loss Functions
        self.criterion_hard = CharbonnierLoss()
        self.criterion_soft = nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def train_setup(self, prm):
        lr = 2e-4 
        weight_decay = 1e-4
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=400, eta_min=1e-6)
        
        # Load Teacher Weights
        teacher_path = "out/ckpt/HSRv4/best_model.pth"
        if os.path.exists(teacher_path):
            state_dict = torch.load(teacher_path, map_location=self.device)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
            self.teacher.load_state_dict(state_dict)
            print(f"✅ [Teacher] Godzilla Loaded from {teacher_path} (KD Ready!)")
        else:
            print(f"❌ [Teacher] WARNING: Weights NOT found at {teacher_path}!")

    def learn(self, data_roll: DataRoll):
        self.current_epoch += 1
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        if self.current_epoch <= 25:
            print(f"--- [STAGE 1] Epoch {self.current_epoch}/25 | Warm-Start L1 Pre-training ---")
        elif self.current_epoch == 26:
            print(f"\n🚀 --- [STAGE 2] Epoch {self.current_epoch} | Godzilla KD Active! ---\n")
            
        for inputs, labels in data_roll:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = torch.clamp(labels, 0.0, 1.0)
            
            self.optimizer.zero_grad()
            student_preds = self.model(inputs)
            
            # STAGE 1: Pure L1 Pre-training (Epochs 1-25)
            if self.current_epoch <= 25:
                loss = self.criterion_soft(student_preds, labels)
                
            # STAGE 2: Output-Level KD (Epochs 26+)
            else:
                with torch.no_grad():
                    teacher_preds = self.teacher(inputs)
                    teacher_preds = torch.clamp(teacher_preds, 0.0, 1.0)
                    
                # 70% Ground Truth + 30% Teacher Output
                loss_hard = self.criterion_hard(student_preds, labels)
                loss_soft_out = self.criterion_soft(student_preds, teacher_preds)
                loss = (0.70 * loss_hard) + (0.30 * loss_soft_out)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            
        self.scheduler.step()
        return total_loss / max(num_batches, 1)
