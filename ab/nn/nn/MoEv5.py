# Enhanced MoE with Diagnostics and Improvements
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import defaultdict

def supported_hyperparameters():
    return {'lr', 'momentum', 'weight_decay', 'lr_schedule'}

class ExpertUtilizationTracker:
    """Track and analyze expert specialization"""
    def __init__(self, n_experts, n_classes=10):
        self.n_experts = n_experts
        self.n_classes = n_classes
        self.reset()
    
    def reset(self):
        self.expert_usage = torch.zeros(self.n_experts)
        self.expert_class_confusion = torch.zeros(self.n_experts, self.n_classes, self.n_classes)
        self.expert_accuracy = torch.zeros(self.n_experts)
        self.expert_samples = torch.zeros(self.n_experts)
        self.total_samples = 0
        
    def update(self, gate_weights, predictions, targets, top_k_indices):
        """Update expert utilization statistics"""
        batch_size = gate_weights.size(0)
        self.total_samples += batch_size
        
        # Track which experts are used
        for batch_idx in range(batch_size):
            active_experts = top_k_indices[batch_idx]
            weights = gate_weights[batch_idx]
            
            for expert_idx in active_experts:
                if expert_idx < self.n_experts:
                    weight = weights[expert_idx].item()
                    self.expert_usage[expert_idx] += weight
                    self.expert_samples[expert_idx] += weight
                    
                    # Track accuracy per expert (weighted)
                    pred = predictions[batch_idx].argmax()
                    target = targets[batch_idx]
                    if pred == target:
                        self.expert_accuracy[expert_idx] += weight
                    
                    # Update confusion matrix for this expert
                    self.expert_class_confusion[expert_idx, target, pred] += weight
    
    def get_specialization_metrics(self):
        """Compute specialization metrics"""
        # Usage distribution entropy (lower = more specialized)
        usage_dist = self.expert_usage / (self.expert_usage.sum() + 1e-8)
        usage_entropy = -(usage_dist * torch.log(usage_dist + 1e-8)).sum().item()
        
        # Individual expert accuracy
        expert_acc = (self.expert_accuracy / (self.expert_samples + 1e-8)).numpy()
        
        # Expert diversity (how different their class preferences are)
        class_preferences = []
        for i in range(self.n_experts):
            if self.expert_samples[i] > 0:
                # Normalize confusion matrix to get class distribution
                class_dist = self.expert_class_confusion[i].sum(dim=1)  # Sum over predictions
                class_dist = class_dist / (class_dist.sum() + 1e-8)
                class_preferences.append(class_dist)
        
        # Compute pairwise Jensen-Shannon divergence between experts
        diversities = []
        if len(class_preferences) >= 2:
            for i in range(len(class_preferences)):
                for j in range(i+1, len(class_preferences)):
                    p, q = class_preferences[i], class_preferences[j]
                    m = 0.5 * (p + q)
                    js_div = 0.5 * F.kl_div(torch.log(p + 1e-8), m, reduction='sum') + \
                            0.5 * F.kl_div(torch.log(q + 1e-8), m, reduction='sum')
                    diversities.append(js_div.item())
        
        return {
            'usage_entropy': usage_entropy,
            'usage_distribution': usage_dist.numpy(),
            'expert_accuracy': expert_acc,
            'expert_samples': self.expert_samples.numpy(),
            'average_diversity': np.mean(diversities) if diversities else 0.0,
            'max_diversity': np.max(diversities) if diversities else 0.0
        }

class ImprovedFeatureExtractor(nn.Module):
    """CIFAR-10 optimized CNN with better efficiency"""
    def __init__(self, in_channels=3):
        super(ImprovedFeatureExtractor, self).__init__()
        
        # Optimized for 32x32 - avoid excessive downsampling
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)  # Keep resolution
        self.bn1 = nn.BatchNorm2d(64)
        
        # Efficient residual blocks with appropriate strides
        self.layer1 = self._make_layer(64, 64, 2, stride=1)    # 32x32 -> 32x32
        self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 32x32 -> 16x16  
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 16x16 -> 8x8
        self.layer4 = self._make_layer(256, 384, 2, stride=2)  # 8x8 -> 4x4
        
        # More aggressive pooling for final features
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))  # 4x4 -> 2x2
        self.dropout = nn.Dropout(0.3)  # Reduced dropout
        
        # Feature dimension: 384 * 2 * 2 = 1536, then reduce
        self.feature_projection = nn.Sequential(
            nn.Linear(384 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)  # Smaller feature dim for faster gating
        )
        self.output_dim = 256

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.feature_projection(x)
        return x

class ExpertWithSpecialization(nn.Module):
    """Smaller, more focused experts"""
    def __init__(self, input_dim, hidden_dim, out_dim, expert_id):
        super(ExpertWithSpecialization, self).__init__()
        self.expert_id = expert_id
        
        # Smaller experts for faster training
        self.fc1 = nn.Linear(input_dim, hidden_dim // 2)  # Reduced size
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, out_dim)
        
        self.ln1 = nn.LayerNorm(hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 4)
        self.dropout = nn.Dropout(0.1)  # Reduced dropout
        
        # Class-specific initialization to encourage specialization
        self._init_class_specific_weights(expert_id)

    def _init_class_specific_weights(self, expert_id):
        """Initialize each expert to be slightly biased toward certain classes"""
        # Different initialization patterns
        patterns = [
            (0.02, 'normal'),      # Expert 0: small normal
            (0.05, 'uniform'),     # Expert 1: larger uniform  
            (1.0, 'xavier'),       # Expert 2: xavier
            (1.0, 'kaiming'),      # Expert 3: kaiming
            (0.1, 'sparse'),       # Expert 4: sparse
            (0.03, 'normal'),      # Expert 5: small normal
            (0.08, 'uniform'),     # Expert 6: larger uniform
            (1.5, 'xavier'),       # Expert 7: larger xavier
        ]
        
        scale, method = patterns[expert_id % len(patterns)]
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == 'normal':
                    nn.init.normal_(module.weight, 0, scale)
                elif method == 'uniform':
                    nn.init.uniform_(module.weight, -scale, scale)
                elif method == 'xavier':
                    nn.init.xavier_normal_(module.weight, gain=scale)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(module.weight, mode='fan_in')
                elif method == 'sparse':
                    nn.init.sparse_(module.weight, sparsity=0.2)
                    
                if module.bias is not None:
                    # Bias the final layer toward specific classes for specialization
                    if module == self.fc3:
                        bias_values = torch.zeros(module.bias.size(0))
                        # Give each expert slight bias toward different class groups
                        preferred_classes = [
                            [0, 1],      # Expert 0: prefer classes 0,1
                            [2, 3],      # Expert 1: prefer classes 2,3  
                            [4, 5],      # Expert 2: prefer classes 4,5
                            [6, 7],      # Expert 3: prefer classes 6,7
                            [8, 9],      # Expert 4: prefer classes 8,9
                            [0, 5],      # Expert 5: mixed preference
                            [1, 6],      # Expert 6: mixed preference  
                            [2, 7],      # Expert 7: mixed preference
                        ]
                        classes = preferred_classes[expert_id % len(preferred_classes)]
                        for cls in classes:
                            if cls < len(bias_values):
                                bias_values[cls] = 0.1  # Small positive bias
                        module.bias.data = bias_values
                    else:
                        nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.gelu(self.ln2(self.fc2(x)))
        x = self.dropout(x)  
        x = self.fc3(x)
        return x

class ImprovedGate(nn.Module):
    """Faster, more decisive gating"""
    def __init__(self, input_dim, n_experts, hidden_dim=128):  # Smaller hidden dim
        super(ImprovedGate, self).__init__()
        self.n_experts = n_experts
        self.top_k = 2
        
        # Simpler but more decisive gating
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_experts)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.05)  # Very light dropout
        
        # Higher temperature for sharper decisions
        self.temperature = nn.Parameter(torch.tensor(2.0))  
        self.noise_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        gate_logits = self.fc2(x)
        
        # Sharper gating decisions
        gate_logits = gate_logits / torch.clamp(self.temperature, min=0.5, max=5.0)
        
        # Less noise during training for more decisive routing
        if self.training:
            noise = torch.randn_like(gate_logits) * torch.clamp(self.noise_scale, min=0.01, max=0.1)
            gate_logits = gate_logits + noise
        
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # More aggressive normalization
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-10)
        
        gates = torch.zeros_like(gate_logits)
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        return gates, top_k_indices, gate_logits

# Add SEBlock and ResidualBlock from original (unchanged)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.n_experts = 8
        self.top_k = 2
        
        # Improved feature extractor 
        self.feature_extractor = ImprovedFeatureExtractor(in_channels=in_shape[1])
        self.feature_dim = self.feature_extractor.output_dim  # Now 256
        
        self.output_dim = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape
        
        # Smaller experts for faster training
        self.hidden_dim = 256  # Reduced from 512
        
        self.experts = nn.ModuleList([
            ExpertWithSpecialization(self.feature_dim, self.hidden_dim, self.output_dim, i)
            for i in range(self.n_experts)
        ])
        
        self.gate = ImprovedGate(self.feature_dim, self.n_experts, 64)  # Smaller gate
        
        # Diagnostic tracker
        self.utilization_tracker = ExpertUtilizationTracker(self.n_experts, self.output_dim)
        
        # Adjusted loss weights for better balance
        self.load_balance_weight = 0.005  # Slightly higher
        self.diversity_weight = 0.002     # Slightly higher
        
        # Reduced label smoothing for CIFAR-10
        self.label_smoothing = 0.05
        
        self.to(device)
        self._print_memory_info()

    def _print_memory_info(self):
        param_count = sum(p.numel() for p in self.parameters())
        param_size_mb = param_count * 4 / (1024 * 1024)
        print(f"Improved MoE-8 Model parameters: {param_count:,}")
        print(f"Model size: {param_size_mb:.2f} MB")
        print(f"Feature dim: {self.feature_dim}, Hidden dim: {self.hidden_dim}")

    def forward(self, x):
        features = self.feature_extractor(x)
        gate_weights, top_k_indices, gate_logits = self.gate(features)
        
        # Efficient expert computation
        expert_outputs = torch.stack([expert(features) for expert in self.experts], dim=2)
        gate_weights_expanded = gate_weights.unsqueeze(1)
        final_output = torch.sum(expert_outputs * gate_weights_expanded, dim=2)
        
        # Update diagnostics during training
        if self.training:
            self._last_gate_weights = gate_weights.detach()
            self._last_top_k_indices = top_k_indices.detach()
            
            # For diversity loss - store only active expert outputs
            active_outputs = []
            unique_experts = torch.unique(top_k_indices)
            for expert_idx in unique_experts[:4]:
                active_outputs.append(expert_outputs[:, :, expert_idx].mean(dim=0).detach())
            self._last_active_outputs = active_outputs
            
        return final_output

    def compute_load_balance_loss(self, gate_weights, expert_indices):
        """Simplified load balancing"""
        if gate_weights is None or gate_weights.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        
        expert_usage = gate_weights.sum(dim=0)
        target_usage = gate_weights.sum() / self.n_experts
        balance_loss = F.mse_loss(expert_usage, target_usage.expand_as(expert_usage))
        
        return torch.clamp(balance_loss, 0.0, 1.0)

    def compute_diversity_loss(self, expert_outputs):
        """Simplified diversity loss"""
        if not expert_outputs or len(expert_outputs) < 2:
            return torch.tensor(0.0, device=self.device)
        
        similarities = []
        for i in range(min(len(expert_outputs), 3)):
            for j in range(i + 1, min(len(expert_outputs), 3)):
                sim = F.cosine_similarity(expert_outputs[i], expert_outputs[j], dim=0)
                similarities.append(sim)
        
        if similarities:
            return torch.stack(similarities).mean()
        return torch.tensor(0.0, device=self.device)

    def train_setup(self, prm):
        self.to(self.device)
        
        self.criteria = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Better optimizer setup for MoE
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=prm.get('lr', 0.002),     # Higher learning rate
            weight_decay=prm.get('weight_decay', 5e-4),  # Higher weight decay
            betas=(0.9, 0.95),           # Different beta2
            eps=1e-6
        )
        
        # Cosine annealing for better convergence
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=5, eta_min=1e-5
        )

    def learn(self, train_data):
        self.train()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        # Reset tracker each epoch
        self.utilization_tracker.reset()
        
        for inputs, labels in train_data:
            inputs = inputs.to(self.device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(self.device, dtype=torch.long, non_blocking=True)

            self.optimizer.zero_grad()
            
            outputs = self(inputs)
            main_loss = self.criteria(outputs, labels)
            
            # Auxiliary losses
            load_loss = torch.tensor(0.0, device=self.device)
            div_loss = torch.tensor(0.0, device=self.device)
            
            if hasattr(self, '_last_gate_weights'):
                load_loss = self.compute_load_balance_loss(
                    self._last_gate_weights, self._last_top_k_indices
                ) * self.load_balance_weight
                
            if hasattr(self, '_last_active_outputs'):
                div_loss = self.compute_diversity_loss(
                    self._last_active_outputs
                ) * self.diversity_weight
            
            total_loss_batch = main_loss + load_loss + div_loss
            total_loss_batch.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            # Track accuracy and expert usage
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update utilization tracker
            if hasattr(self, '_last_gate_weights'):
                self.utilization_tracker.update(
                    self._last_gate_weights, predicted, labels, self._last_top_k_indices
                )
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            # Memory cleanup
            if hasattr(self, '_last_gate_weights'):
                delattr(self, '_last_gate_weights')
            if hasattr(self, '_last_top_k_indices'):
                delattr(self, '_last_top_k_indices')  
            if hasattr(self, '_last_active_outputs'):
                delattr(self, '_last_active_outputs')
        
        # Update scheduler
        self.scheduler.step()
        
        # Print diagnostics
        metrics = self.utilization_tracker.get_specialization_metrics()
        train_acc = 100. * correct / total
        
        print(f"Epoch Training Accuracy: {train_acc:.2f}%")
        print(f"Expert Usage Entropy: {metrics['usage_entropy']:.3f} (lower = more specialized)")
        print(f"Expert Usage Distribution: {metrics['usage_distribution']}")
        print(f"Expert Diversity (JS): {metrics['average_diversity']:.3f} (higher = more diverse)")
        print(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        return total_loss / num_batches if num_batches > 0 else 0