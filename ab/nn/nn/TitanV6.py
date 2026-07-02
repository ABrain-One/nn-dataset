import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from ab.nn.util.Classes import DataRoll
from ab.nn.nn.HSRv4 import HSR_Godzilla

# Empty set disables Optuna's random LR overriding
def supported_hyperparameters():
    return set()

# --- 1. ARCHITECTURE: CALayer & EdgeConvBlock ---
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class EdgeConvBlock(nn.Module):
    def __init__(self, channels):
        super(EdgeConvBlock, self).__init__()
        self.conv_3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_1x1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv_1x3 = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1))
        self.conv_3x1 = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0))
        self.identity = nn.Identity()
        self.attention = CALayer(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_3x3 = self.conv_3x3(x)
        out_1x1 = self.conv_1x1(x)
        out_1x3 = self.conv_1x3(x)
        out_3x1 = self.conv_3x1(x)
        out_id = self.identity(x)
        out = out_3x3 + out_1x1 + out_1x3 + out_3x1 + out_id
        out = self.attention(out)
        return self.relu(out)

# --- 2. UPGRADED TITAN V6 MODEL ---
class TitanV6(nn.Module):
    def __init__(self, channels=48):
        super(TitanV6, self).__init__()
        self.entry = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.block1 = EdgeConvBlock(channels)
        self.block2 = EdgeConvBlock(channels)
        self.block3 = EdgeConvBlock(channels)
        self.block4 = EdgeConvBlock(channels)
        self.block5 = EdgeConvBlock(channels)
        self.block6 = EdgeConvBlock(channels)
        self.exit = nn.Conv2d(channels, 27, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(3)

    def forward(self, x):
        out_entry = self.entry(x)
        out = self.block1(out_entry)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = out + out_entry 
        out = self.exit(out)
        return self.pixel_shuffle(out)

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
        
        # 1. Student Model (TitanV6 with 6 EdgeConvBlocks, 48 channels)
        self.model = TitanV6(channels=48).to(device)
        
        # Feature KD Setup (Adapter & Hooks Dict)
        self.feature_adapter = nn.Conv2d(48, 180, 1).to(device)
        self.features = {'student': None, 'teacher': None}
        
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
        # --- WARM RESTART SETUP ---
        lr = 5e-5  # Reduced LR for fine-tuning
        weight_decay = 1e-4
        
        # 1. Load Saved Student Weights (Warm Restart)
        student_path = "out/ckpt/TitanV5/best_model.pth"
        if os.path.exists(student_path):
            state_dict = torch.load(student_path, map_location=self.device)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.') or not k.startswith('teacher')}
            self.model.load_state_dict(state_dict, strict=False)
            print(f"🔥 [Warm Restart] TitanV6 Loaded from {student_path} | Ready for Fine-tuning!")
        else:
            print(f"⚠️ [Warm Restart] WARNING: Student weights NOT found at {student_path}!")

        # 2. Fresh Adam Optimizer & Scheduler (Include Adapter)
        trainable_params = list(self.model.parameters()) + list(self.feature_adapter.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=400, eta_min=1e-6)
        
        # 3. Reset Epoch
        self.current_epoch = 0

        # 4. PyTorch Forward Hooks Setup
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
            
        self.model.block6.register_forward_hook(get_features('student'))
        self.teacher.tail_conv.register_forward_hook(get_features('teacher'))

        # 5. Load Teacher Weights (Hamesha ki tarah)
        teacher_path = "out/ckpt/HSRv4/best_model.pth"
        if os.path.exists(teacher_path):
            state_dict = torch.load(teacher_path, map_location=self.device)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
            self.teacher.load_state_dict(state_dict)
            print(f"✅ [Teacher] Godzilla Loaded from {teacher_path} (KD Ready!)")
        else:
            print(f"❌ [Teacher] WARNING: Weights NOT found at {teacher_path}!")

    # --- 3. TRAINING LOOP (WARM-UP + KD) ---
    def learn(self, data_roll: DataRoll):
        self.current_epoch += 1
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        if self.current_epoch <= 5:
            print(f"--- [STAGE 1] Epoch {self.current_epoch}/5 | Stretching & Warm-up (L1) ---")
        elif self.current_epoch == 6:
            print(f"\n🚀 --- [STAGE 2] Epoch {self.current_epoch} | Godzilla KD Active! ---\n")
            
        for inputs, labels in data_roll:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = torch.clamp(labels, 0.0, 1.0)
            
            # --- CutBlur Augmentation ---
            if random.random() < 0.5:
                _, _, lr_h, lr_w = inputs.size()
                
                w = int(random.uniform(0.25, 0.5) * lr_w)
                h = int(random.uniform(0.25, 0.5) * lr_h)
                
                x0 = random.randint(0, lr_w - w)
                y0 = random.randint(0, lr_h - h)
                
                scale = 3
                hr_x0, hr_y0, hr_w, hr_h = x0 * scale, y0 * scale, w * scale, h * scale
                
                lr_patch = inputs[:, :, y0:y0+h, x0:x0+w].clone()
                hr_patch = labels[:, :, hr_y0:hr_y0+hr_h, hr_x0:hr_x0+hr_w].clone()
                
                upsampled_lr = F.interpolate(lr_patch, size=(hr_h, hr_w), mode='bicubic', align_corners=False)
                labels[:, :, hr_y0:hr_y0+hr_h, hr_x0:hr_x0+hr_w] = upsampled_lr
                
                downsampled_hr = F.interpolate(hr_patch, size=(h, w), mode='bicubic', align_corners=False)
                inputs[:, :, y0:y0+h, x0:x0+w] = downsampled_hr
            # ---------------------------
            
            # Clear Hook Dictionary Memory Leak
            self.features['student'] = None
            self.features['teacher'] = None
            
            self.optimizer.zero_grad()
            student_preds = self.model(inputs)
            
            # STAGE 1: Pure L1 Pre-training (Epochs 1-5)
            if self.current_epoch <= 5:
                loss = self.criterion_soft(student_preds, labels) # Standard L1 Loss
                
            # STAGE 2: Heavy KD Formula (Epochs 6+)
            else:
                with torch.no_grad():
                    teacher_preds = self.teacher(inputs)
                    teacher_preds = torch.clamp(teacher_preds, 0.0, 1.0)
                    
                loss_charbonnier = self.criterion_hard(student_preds, labels)
                loss_distill = self.criterion_soft(student_preds, teacher_preds)
                loss_l1 = self.criterion_soft(student_preds, labels)
                
                # Feature-Level Loss Projection
                student_features_mapped = self.feature_adapter(self.features['student'])
                loss_feature = self.criterion_soft(student_features_mapped, self.features['teacher'].detach())
                
                # The 29dB Winning Formula (Normalized + Feature KD)
                loss = (1.0 * loss_charbonnier) + (9.0 * loss_distill) + (0.5 * loss_l1) + (1.0 * loss_feature)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.feature_adapter.parameters()), 10.0)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            
        self.scheduler.step()
        return total_loss / max(num_batches, 1)

    # --- 4. VALIDATION LOOP (DR. IGNATOV SCRIPT SIMULATOR) ---
    def evaluate(self, data_roll: DataRoll):
        self.model.eval()
        total_psnr = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, labels in data_roll:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                student_preds = self.model(inputs)
                
                # -----------------------------------------------------
                # 🔥 THE REAL-WORLD SCRIPT SIMULATOR 🔥
                # Yeh hissa PyTorch ke high-precision illusion ko tod kar 
                # mobile NPU jaisa exact behavior layega.
                # -----------------------------------------------------
                preds_255 = student_preds * 255.0
                preds_rounded = torch.round(preds_255)
                preds_clamped = torch.clamp(preds_rounded, 0.0, 255.0)
                final_preds = preds_clamped / 255.0
                
                labels_clamped = torch.clamp(labels * 255.0, 0.0, 255.0) / 255.0
                
                # Yahan aap apna normal PSNR calculate karein final_preds aur labels_clamped par
                # psnr_val = calculate_psnr(final_preds, labels_clamped)
                # total_psnr += psnr_val
                num_batches += 1
                
        # return total_psnr / max(num_batches, 1)
