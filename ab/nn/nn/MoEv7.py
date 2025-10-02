# MoE-v7: Advanced Architecture for 90%+ Accuracy with Comprehensive Diagnostics
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import math


def supported_hyperparameters():
    return {'lr', 'momentum', 'weight_decay', 'warmup_epochs', 'mixup_alpha'}


class ComprehensiveDiagnostics:
    """Advanced diagnostic system for MoE analysis"""
    def __init__(self, n_experts, n_classes=10):
        self.n_experts = n_experts
        self.n_classes = n_classes
        self.reset()
        
    def reset(self):
        """Reset all metrics for new epoch"""
        self.expert_usage = torch.zeros(self.n_experts)
        self.expert_confidence = torch.zeros(self.n_experts)
        self.expert_correct = torch.zeros(self.n_experts)
        self.expert_total = torch.zeros(self.n_experts)
        self.expert_class_predictions = torch.zeros(self.n_experts, self.n_classes)
        self.expert_class_correct = torch.zeros(self.n_experts, self.n_classes)
        self.gate_weights_history = []
        self.routing_entropy_history = []
        self.batch_count = 0
        
    def update(self, gate_weights, predictions, targets, outputs):
        """Update diagnostics with batch results"""
        self.batch_count += 1
        batch_size = gate_weights.size(0)
        
        # Expert usage statistics
        self.expert_usage += gate_weights.sum(dim=0).cpu()
        
        # Confidence and accuracy per expert
        for batch_idx in range(batch_size):
            pred = predictions[batch_idx]
            target = targets[batch_idx]
            
            # Weight each expert's contribution by gate weight
            for expert_idx in range(self.n_experts):
                weight = gate_weights[batch_idx, expert_idx].item()
                if weight > 0.01:  # Only count if expert is actually used
                    self.expert_total[expert_idx] += weight
                    self.expert_confidence[expert_idx] += weight * outputs[batch_idx, pred].item()
                    
                    # Track predictions
                    self.expert_class_predictions[expert_idx, pred] += weight
                    
                    # Track correctness
                    if pred == target:
                        self.expert_correct[expert_idx] += weight
                        self.expert_class_correct[expert_idx, target] += weight
        
        # Routing entropy (measure of gate decisiveness)
        gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=1).mean()
        self.routing_entropy_history.append(gate_entropy.item())
        
        # Store sample of gate weights for analysis
        if len(self.gate_weights_history) < 10:
            self.gate_weights_history.append(gate_weights[:min(8, batch_size)].detach().cpu())
    
    def get_report(self):
        """Generate comprehensive diagnostic report"""
        report = {}
        
        # Expert utilization
        usage_dist = self.expert_usage / (self.expert_usage.sum() + 1e-8)
        usage_entropy = -(usage_dist * torch.log(usage_dist + 1e-8)).sum().item()
        report['usage_distribution'] = usage_dist.numpy()
        report['usage_entropy'] = usage_entropy
        report['usage_std'] = usage_dist.std().item()
        
        # Expert accuracy
        expert_acc = (self.expert_correct / (self.expert_total + 1e-8)).numpy()
        report['expert_accuracy'] = expert_acc
        report['expert_accuracy_mean'] = expert_acc.mean()
        report['expert_accuracy_std'] = expert_acc.std()
        
        # Expert confidence
        expert_conf = (self.expert_confidence / (self.expert_total + 1e-8)).numpy()
        report['expert_confidence'] = expert_conf
        
        # Class specialization
        class_prefs = self.expert_class_predictions / (self.expert_class_predictions.sum(dim=1, keepdim=True) + 1e-8)
        # Calculate specialization score (how peaked is each expert's class distribution)
        specialization_scores = -(class_prefs * torch.log(class_prefs + 1e-8)).sum(dim=1).numpy()
        report['specialization_scores'] = specialization_scores
        report['avg_specialization'] = specialization_scores.mean()
        
        # Routing entropy
        report['avg_routing_entropy'] = np.mean(self.routing_entropy_history)
        
        return report
    
    def print_report(self, epoch):
        """Print formatted diagnostic report"""
        report = self.get_report()
        
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch} DIAGNOSTICS")
        print(f"{'='*70}")
        
        print(f"\n📊 Expert Usage:")
        print(f"   Distribution: {report['usage_distribution']}")
        print(f"   Entropy: {report['usage_entropy']:.4f} (lower=more specialized)")
        print(f"   Std Dev: {report['usage_std']:.4f}")
        
        print(f"\n🎯 Expert Performance:")
        for i, (acc, conf) in enumerate(zip(report['expert_accuracy'], report['expert_confidence'])):
            print(f"   Expert {i}: Acc={acc:.4f}, Confidence={conf:.4f}")
        print(f"   Mean Accuracy: {report['expert_accuracy_mean']:.4f} ± {report['expert_accuracy_std']:.4f}")
        
        print(f"\n🔬 Specialization:")
        print(f"   Scores: {report['specialization_scores']}")
        print(f"   Average: {report['avg_specialization']:.4f} (lower=more specialized)")
        
        print(f"\n🔀 Routing:")
        print(f"   Avg Entropy: {report['avg_routing_entropy']:.4f}")
        
        print(f"{'='*70}\n")


class DualPathBlock(nn.Module):
    """Dual-path residual block with both spatial and channel attention"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Squeeze-and-Excitation (channel attention)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply SE attention
        se_weight = self.se(out)
        out = out * se_weight
        
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.dropout(out)
        
        return out


class AdvancedFeatureExtractor(nn.Module):
    """State-of-the-art CNN for CIFAR-10 with multiple enhancements"""
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Initial convolution - preserve resolution
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Progressive feature extraction with dual-path blocks
        self.layer1 = self._make_layer(64, 64, 3, stride=1)    # 32x32
        self.layer2 = self._make_layer(64, 128, 3, stride=2)   # 16x16
        self.layer3 = self._make_layer(128, 256, 3, stride=2)  # 8x8
        self.layer4 = self._make_layer(256, 512, 3, stride=2)  # 4x4
        
        # Global context
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(512 * 2, 384),  # Concatenate avg and max pooling
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, 256)
        )
        
        self.output_dim = 256
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(DualPathBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(DualPathBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Dual pooling for richer features
        avg_pool = self.avgpool(x).flatten(1)
        max_pool = self.maxpool(x).flatten(1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        x = self.feature_proj(x)
        return x


class AdaptiveExpert(nn.Module):
    """Expert with adaptive capacity and specialized initialization"""
    def __init__(self, input_dim, hidden_dim, out_dim, expert_id, n_experts):
        super().__init__()
        self.expert_id = expert_id
        
        # 4-layer expert with residual connections
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, out_dim)
        
        # Layer normalization for stable training
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        
        # Adaptive dropout
        self.dropout = nn.Dropout(0.15)
        
        # Residual connection
        self.residual = nn.Linear(input_dim, hidden_dim)
        
        # Initialize for specialization
        self._init_specialized_weights(expert_id, n_experts)
    
    def _init_specialized_weights(self, expert_id, n_experts):
        """Initialize each expert differently to encourage specialization"""
        # Use different initialization schemes
        init_schemes = [
            lambda m: nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu'),
            lambda m: nn.init.xavier_uniform_(m.weight, gain=1.414),
            lambda m: nn.init.orthogonal_(m.weight, gain=1.0),
            lambda m: nn.init.xavier_normal_(m.weight, gain=1.0),
            lambda m: nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu'),
        ]
        
        scheme = init_schemes[expert_id % len(init_schemes)]
        
        for module in [self.fc1, self.fc2, self.fc3]:
            scheme(module)
        
        # Final layer: bias toward specific classes
        nn.init.xavier_uniform_(self.fc4.weight)
        if self.fc4.bias is not None:
            # Each expert gets slight bias toward different class groups
            bias = torch.zeros_like(self.fc4.bias)
            classes_per_expert = 10 // n_experts
            start_class = (expert_id * classes_per_expert) % 10
            end_class = (start_class + classes_per_expert + 1) % 10
            
            if start_class < end_class:
                bias[start_class:end_class] = 0.1
            else:
                bias[start_class:] = 0.1
                bias[:end_class] = 0.1
            
            self.fc4.bias.data = bias
    
    def forward(self, x):
        # Residual connection from input
        residual = self.residual(x)
        
        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        
        # Add residual
        x = x + residual
        
        x = F.gelu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.gelu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x


class DynamicGate(nn.Module):
    """Advanced gating with learned routing strategies"""
    def __init__(self, input_dim, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        # Multi-head gating for diverse routing perspectives
        self.n_heads = 4
        head_dim = 64
        
        self.input_proj = nn.Linear(input_dim, head_dim * self.n_heads)
        
        # Per-head gating
        self.gate_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_dim, head_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(head_dim, n_experts)
            )
            for _ in range(self.n_heads)
        ])
        
        # Combine heads
        self.head_combiner = nn.Linear(n_experts * self.n_heads, n_experts)
        
        # Learnable parameters
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)
        self.noise_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        # Project input
        x = self.input_proj(x)
        x = x.view(-1, self.n_heads, x.size(-1) // self.n_heads)
        
        # Multi-head gating
        head_logits = []
        for i, gate_head in enumerate(self.gate_heads):
            head_logits.append(gate_head(x[:, i, :]))
        
        # Combine heads
        combined_logits = torch.cat(head_logits, dim=-1)
        gate_logits = self.head_combiner(combined_logits)
        
        # Temperature scaling
        gate_logits = gate_logits / torch.clamp(self.temperature, 0.5, 5.0)
        
        # Training noise for exploration
        if self.training:
            noise = torch.randn_like(gate_logits) * torch.clamp(self.noise_scale, 0.01, 0.2)
            gate_logits = gate_logits + noise
        
        # Top-k routing
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Sparse representation
        gates = torch.zeros_like(gate_logits)
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        return gates, top_k_indices, gate_logits


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.n_experts = 10  # Increased for better capacity
        self.top_k = 2
        
        # Advanced feature extractor
        self.feature_extractor = AdvancedFeatureExtractor(in_channels=in_shape[1])
        self.feature_dim = self.feature_extractor.output_dim
        
        self.output_dim = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape
        self.hidden_dim = 384
        
        # Create adaptive experts
        self.experts = nn.ModuleList([
            AdaptiveExpert(self.feature_dim, self.hidden_dim, self.output_dim, i, self.n_experts)
            for i in range(self.n_experts)
        ])
        
        # Dynamic gating
        self.gate = DynamicGate(self.feature_dim, self.n_experts, self.top_k)
        
        # Auxiliary loss weights
        self.load_balance_weight = 0.008
        self.diversity_weight = 0.003
        
        # Label smoothing
        self.label_smoothing = 0.1
        
        # Mixup augmentation
        self.mixup_alpha = prm.get('mixup_alpha', 0.2)
        
        # Diagnostics
        self.diagnostics = ComprehensiveDiagnostics(self.n_experts, self.output_dim)
        
        self.to(device)
        self._print_info()
    
    def _print_info(self):
        param_count = sum(p.numel() for p in self.parameters())
        param_size_mb = param_count * 4 / (1024 * 1024)
        print(f"\n{'='*70}")
        print(f"MoE-v7 ADVANCED MODEL")
        print(f"{'='*70}")
        print(f"Parameters: {param_count:,} ({param_size_mb:.2f} MB)")
        print(f"Experts: {self.n_experts} | Top-K: {self.top_k}")
        print(f"Feature dim: {self.feature_dim} | Hidden dim: {self.hidden_dim}")
        print(f"Architecture: DualPath + Multi-head Gating + Adaptive Experts")
        print(f"{'='*70}\n")
    
    def mixup_data(self, x, y, alpha=0.2):
        """Mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def forward(self, x):
        features = self.feature_extractor(x)
        gate_weights, top_k_indices, gate_logits = self.gate(features)
        
        # Compute expert outputs
        expert_outputs = torch.stack([expert(features) for expert in self.experts], dim=2)
        
        # Weighted combination
        gate_weights_expanded = gate_weights.unsqueeze(1)
        final_output = torch.sum(expert_outputs * gate_weights_expanded, dim=2)
        
        # Store for diagnostics
        if self.training:
            self._last_gate_weights = gate_weights.detach()
            self._last_top_k_indices = top_k_indices.detach()
            self._last_expert_outputs = expert_outputs.detach()
        
        return final_output
    
    def compute_auxiliary_losses(self):
        """Compute load balance and diversity losses"""
        if not hasattr(self, '_last_gate_weights'):
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        
        gate_weights = self._last_gate_weights
        
        # Load balance loss
        expert_usage = gate_weights.sum(dim=0)
        target_usage = gate_weights.sum() / self.n_experts
        load_loss = F.mse_loss(expert_usage, target_usage.expand_as(expert_usage))
        
        # Diversity loss - encourage experts to produce different outputs
        expert_outputs = self._last_expert_outputs
        similarities = []
        for i in range(min(4, self.n_experts)):
            for j in range(i + 1, min(4, self.n_experts)):
                output_i = expert_outputs[:, :, i].mean(dim=0)
                output_j = expert_outputs[:, :, j].mean(dim=0)
                sim = F.cosine_similarity(output_i.unsqueeze(0), output_j.unsqueeze(0))
                similarities.append(sim)
        
        diversity_loss = torch.stack(similarities).mean() if similarities else torch.tensor(0.0, device=self.device)
        
        return load_loss, diversity_loss
    
    def train_setup(self, prm):
        self.to(self.device)
        
        self.criteria = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm.get('lr', 0.001),
            weight_decay=prm.get('weight_decay', 5e-4),
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warmup
        warmup_epochs = prm.get('warmup_epochs', 5)
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        self.main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        self.current_epoch = 0
        self.warmup_epochs = warmup_epochs
    
    def learn(self, train_data):
        self.train()
        self.diagnostics.reset()
        
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        for inputs, labels in train_data:
            inputs = inputs.to(self.device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
            
            # Mixup augmentation
            if self.training and self.mixup_alpha > 0:
                inputs, labels_a, labels_b, lam = self.mixup_data(inputs, labels, self.mixup_alpha)
                
                self.optimizer.zero_grad()
                outputs = self(inputs)
                
                # Mixup loss
                loss = lam * self.criteria(outputs, labels_a) + (1 - lam) * self.criteria(outputs, labels_b)
            else:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criteria(outputs, labels)
            
            # Auxiliary losses
            load_loss, div_loss = self.compute_auxiliary_losses()
            total_loss_batch = loss + self.load_balance_weight * load_loss + self.diversity_weight * div_loss
            
            total_loss_batch.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item()
            num_batches += 1
            
            # Update diagnostics
            if hasattr(self, '_last_gate_weights'):
                self.diagnostics.update(
                    self._last_gate_weights,
                    predicted.cpu(),
                    labels.cpu(),
                    outputs.detach().cpu()
                )
        
        # Update scheduler
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.main_scheduler.step()
        
        # Print diagnostics
        train_acc = 100. * correct / total
        print(f"\n📈 Training Accuracy: {train_acc:.2f}%")
        print(f"📉 Average Loss: {total_loss / num_batches:.4f}")
        print(f"🔧 Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        self.diagnostics.print_report(self.current_epoch)
        
        return total_loss / num_batches