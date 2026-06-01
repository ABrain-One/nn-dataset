import os
import torch
import torch.nn as nn
from ab.nn.nn.HSRv4 import HSR_Godzilla
from ab.nn.nn.TitanV5 import TitanV5
from ab.nn.util.Classes import DataRoll

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.scale = 3
        
        # 1. Student (TitanV5_Lite - 36 Channels, 4 Blocks)
        # Note: TitanV5 internally uses nearest interpolation as set previously
        self.model = TitanV5(f=36, n_blocks=4, upscale=self.scale).to(device)
        
        # 2. Teacher (Godzilla - 180 Channels)
        # We define it exactly as it was trained to match the 31MB weights
        self.teacher = HSR_Godzilla(
            feature_channels=180, 
            n_groups=5, 
            n_blocks=8, 
            upscale=self.scale
        ).to(device)
        
        self.criterion = nn.L1Loss()
        
        # Hyperparameters
        self.kd_weight = prm.get('kd_weight', 1.0)
        self.lr = prm.get('lr', 1e-4) # Safe LR for RepConv

    def train_setup(self, prm):
        """Setup optimizers, schedulers, and load teacher weights."""
        # 1. Teacher Weights Loading
        teacher_path = "out/ckpt/HSRv4/best_model.pth"
        if os.path.exists(teacher_path):
            state_dict = torch.load(teacher_path, map_location=self.device)
            # The weights were saved using a Net wrapper, so keys have 'model.' prefix
            # Since our teacher is HSR_Godzilla directly, we need to strip 'model.' prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
                    
            self.teacher.load_state_dict(new_state_dict, strict=True)
            print(f"💎 [KD SETUP] Teacher (Godzilla) Weights Loaded Successfully!")
        else:
            print(f"❌ [KD SETUP] Teacher weights not found at {teacher_path}!")
            
        # Freeze Teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        # 2. Student Optimizer & Scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Cosine Annealing scheduler (T_max = EPOCHS)
        epochs = prm.get('epoch', 200)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

    def learn(self, data_roll: DataRoll):
        """Framework training loop for KD."""
        self.model.train()
        total_loss = 0.0
        
        for inputs, labels in data_roll:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward Passes
            out_student = self.model(inputs)
            with torch.no_grad():
                out_teacher = self.teacher(inputs)
                
            # Distillation Loss
            loss_pix = self.criterion(out_student, labels)
            loss_kd = self.criterion(out_student, out_teacher)
            loss = loss_pix + (self.kd_weight * loss_kd)
            
            loss.backward()
            
            # Gradient clipping to protect RepConv blocks from explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        # Step the scheduler after each epoch
        self.scheduler.step()
        
        # Returning loss to framework (Optional, depending on if it expects it)
        return total_loss

    def forward(self, x):
        """Standard forward pass for Validation and Evaluation (Student Only)."""
        return self.model(x)
