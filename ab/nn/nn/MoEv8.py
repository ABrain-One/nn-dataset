# MoE-v7-Fixed: Addressing Expert Collapse with Proper Load Balancing
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import math


def supported_hyperparameters():
    return {'lr', 'weight_decay', 'warmup_epochs', 'mixup_alpha'}


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


class WideResidualBlock(nn.Module):
    """Wide residual block for better feature capacity"""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super().__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        self.dropout = nn.Dropout2d(dropout)
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
    
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        
        out += self.shortcut(x)
        
        return out


class ImprovedFeatureExtractor(nn.Module):
    """WideResNet-inspired CNN for CIFAR-10"""
    def __init__(self, in_channels=3, widen_factor=4):
        super().__init__()
        
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        
        self.conv1 = nn.Conv2d(in_channels, nChannels[0], 3, padding=1, bias=False)
        
        # 3 groups of residual blocks
        self.block1 = self._make_layer(WideResidualBlock, nChannels[0], nChannels[1], 4, stride=1)
        self.block2 = self._make_layer(WideResidualBlock, nChannels[1], nChannels[2], 4, stride=2)
        self.block3 = self._make_layer(WideResidualBlock, nChannels[2], nChannels[3], 4, stride=2)
        
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.output_dim = nChannels[3]
    
    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out


class BalancedExpert(nn.Module):
    """Simpler expert to prevent overfitting"""
    def __init__(self, input_dim, hidden_dim, out_dim, expert_id):
        super().__init__()
        self.expert_id = expert_id
        
        # 3-layer expert
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, out_dim)
        
        self.dropout = nn.Dropout(0.2)
        
        # Initialize with small random values to encourage diversity
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class BalancedGate(nn.Module):
    """Gating network with strong load balancing"""
    def __init__(self, input_dim, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        # Simple but effective gating
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_experts)
        )
        
        # Learnable temperature - start higher to encourage exploration
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        # Higher noise for better load balancing
        self.noise_std = 0.5
    
    def forward(self, x):
        gate_logits = self.gate_net(x)
        
        # Temperature scaling - clamp to reasonable range
        gate_logits = gate_logits / torch.clamp(self.temperature, 0.5, 2.0)
        
        # Add significant noise during training for load balancing
        if self.training:
            # Stronger noise to force exploration
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
        
        # Top-k routing
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Create sparse gates
        gates = torch.zeros_like(gate_logits)
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        return gates, top_k_indices, gate_logits


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.n_experts = 8  # Reduced from 10 for better balance
        self.top_k = 2
        
        # WideResNet feature extractor
        self.feature_extractor = ImprovedFeatureExtractor(in_channels=in_shape[1], widen_factor=4)
        self.feature_dim = self.feature_extractor.output_dim
        
        self.output_dim = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape
        self.hidden_dim = 256  # Reduced to prevent overfitting
        
        # Create balanced experts
        self.experts = nn.ModuleList([
            BalancedExpert(self.feature_dim, self.hidden_dim, self.output_dim, i)
            for i in range(self.n_experts)
        ])
        
        # Balanced gating
        self.gate = BalancedGate(self.feature_dim, self.n_experts, self.top_k)
        
        # MUCH STRONGER auxiliary loss weights to force balance
        self.load_balance_weight = 0.1  # 10x stronger
        self.importance_weight = 0.05   # New: importance loss
        
        # Label smoothing
        self.label_smoothing = 0.1
        
        # Mixup
        self.mixup_alpha = prm.get('mixup_alpha', 0.4)
        
        # Diagnostics
        self.diagnostics = ComprehensiveDiagnostics(self.n_experts, self.output_dim)
        
        self.to(device)
        self._print_info()
    
    def _print_info(self):
        param_count = sum(p.numel() for p in self.parameters())
        param_size_mb = param_count * 4 / (1024 * 1024)
        print(f"\n{'='*70}")
        print(f"MoE-v7-FIXED MODEL (Balanced Training)")
        print(f"{'='*70}")
        print(f"Parameters: {param_count:,} ({param_size_mb:.2f} MB)")
        print(f"Experts: {self.n_experts} | Top-K: {self.top_k}")
        print(f"Feature dim: {self.feature_dim} | Hidden dim: {self.hidden_dim}")
        print(f"Load Balance Weight: {self.load_balance_weight} (STRONG)")
        print(f"Architecture: WideResNet + Balanced Gating")
        print(f"{'='*70}\n")
    
    def mixup_data(self, x, y, alpha=0.4):
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
        
        # Store for diagnostics and losses
        if self.training:
            self._last_gate_weights = gate_weights
            self._last_gate_logits = gate_logits
            self._last_expert_outputs = expert_outputs
        
        return final_output
    
    def compute_auxiliary_losses(self):
        """Strong auxiliary losses to enforce load balancing"""
        if not hasattr(self, '_last_gate_weights'):
            return torch.tensor(0.0, device=self.device)
        
        gate_weights = self._last_gate_weights
        
        # 1. Load balance loss - L2 penalty on uneven distribution
        expert_usage = gate_weights.sum(dim=0)
        target = expert_usage.sum() / self.n_experts
        load_loss = ((expert_usage - target) ** 2).mean()
        
        # 2. Importance loss - from Switch Transformer paper
        # Encourages all experts to have similar importance
        importance = gate_weights.sum(dim=0)
        importance_loss = importance.var()
        
        # 3. Entropy regularization - encourage uncertainty in routing
        gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=1).mean()
        entropy_loss = -gate_entropy  # Maximize entropy (minimize negative entropy)
        
        total_aux_loss = (
            self.load_balance_weight * load_loss +
            self.importance_weight * importance_loss +
            0.01 * entropy_loss
        )
        
        return total_aux_loss
    
    def train_setup(self, prm):
        self.to(self.device)
        
        self.criteria = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Use SGD with strong momentum for stability
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.1),  # Higher initial LR
            momentum=0.9,
            weight_decay=prm.get('weight_decay', 5e-4),
            nesterov=True
        )
        
        # MultiStepLR for step-wise decay
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[60, 120, 160],
            gamma=0.2
        )
        
        self.current_epoch = 0
    
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
            if self.training and np.random.rand() < 0.5:  # 50% of the time
                inputs, labels_a, labels_b, lam = self.mixup_data(inputs, labels, self.mixup_alpha)
                
                self.optimizer.zero_grad()
                outputs = self(inputs)
                
                # Mixup loss
                loss = lam * self.criteria(outputs, labels_a) + (1 - lam) * self.criteria(outputs, labels_b)
                
                # Use original labels for metrics
                labels_for_metrics = labels_a
            else:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criteria(outputs, labels)
                labels_for_metrics = labels
            
            # STRONG auxiliary losses
            aux_loss = self.compute_auxiliary_losses()
            total_loss_batch = loss + aux_loss
            
            total_loss_batch.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            _, predicted = outputs.max(1)
            total += labels_for_metrics.size(0)
            correct += predicted.eq(labels_for_metrics).sum().item()
            total_loss += loss.item()
            num_batches += 1
            
            # Update diagnostics
            if hasattr(self, '_last_gate_weights'):
                self.diagnostics.update(
                    self._last_gate_weights.detach(),
                    predicted.cpu(),
                    labels_for_metrics.cpu(),
                    outputs.detach().cpu()
                )
        
        # Update scheduler
        self.scheduler.step()
        self.current_epoch += 1
        
        # Print diagnostics
        train_acc = 100. * correct / total
        print(f"\n📈 Training Accuracy: {train_acc:.2f}%")
        print(f"📉 Average Loss: {total_loss / num_batches:.4f}")
        print(f"🔧 Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        self.diagnostics.print_report(self.current_epoch)
        
        return total_loss / num_batches