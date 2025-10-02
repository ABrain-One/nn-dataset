# Enhanced MoE with AlexNet Experts
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import defaultdict

def supported_hyperparameters():
    return {'lr', 'momentum', 'weight_decay', 'dropout', 'lr_schedule'}

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
                class_dist = self.expert_class_confusion[i].sum(dim=1)
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


class AlexNetExpert(nn.Module):
    """AlexNet architecture adapted as an expert in MoE"""
    def __init__(self, in_channels, out_dim, dropout, expert_id):
        super(AlexNetExpert, self).__init__()
        self.expert_id = expert_id
        
        # AlexNet feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # AlexNet classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_dim),
        )
        
        # Class-specific initialization to encourage specialization
        self._init_class_specific_weights(expert_id, out_dim)
    
    def _init_class_specific_weights(self, expert_id, out_dim):
        """Initialize each expert to be slightly biased toward certain classes"""
        # Apply different initialization strategies per expert
        patterns = [
            (0.02, 'normal'),
            (0.05, 'uniform'),
            (1.0, 'xavier'),
            (1.0, 'kaiming'),
            (0.1, 'sparse'),
            (0.03, 'normal'),
            (0.08, 'uniform'),
            (1.5, 'xavier'),
        ]
        
        scale, method = patterns[expert_id % len(patterns)]
        
        # Apply to final classifier layer only for specialization
        final_layer = self.classifier[-1]
        if method == 'normal':
            nn.init.normal_(final_layer.weight, 0, scale)
        elif method == 'uniform':
            nn.init.uniform_(final_layer.weight, -scale, scale)
        elif method == 'xavier':
            nn.init.xavier_normal_(final_layer.weight, gain=scale)
        elif method == 'kaiming':
            nn.init.kaiming_normal_(final_layer.weight, mode='fan_in')
        elif method == 'sparse':
            nn.init.sparse_(final_layer.weight, sparsity=0.2)
        
        # Bias the final layer toward specific classes for specialization
        if final_layer.bias is not None:
            bias_values = torch.zeros(out_dim)
            preferred_classes = [
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9],
                [0, 5],
                [1, 6],
                [2, 7],
            ]
            classes = preferred_classes[expert_id % len(preferred_classes)]
            for cls in classes:
                if cls < out_dim:
                    bias_values[cls] = 0.1
            final_layer.bias.data = bias_values
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ImprovedGate(nn.Module):
    """Faster, more decisive gating network"""
    def __init__(self, input_channels, n_experts, hidden_dim=128):
        super(ImprovedGate, self).__init__()
        self.n_experts = n_experts
        self.top_k = 2
        
        # Lightweight CNN to process input images for routing
        self.gate_features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # MLP for gating decision
        self.fc1 = nn.Linear(64 * 4 * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_experts)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.05)
        
        # Temperature for sharpness
        self.temperature = nn.Parameter(torch.tensor(2.0))
        self.noise_scale = nn.Parameter(torch.tensor(0.05))
    
    def forward(self, x):
        # Extract features for gating
        gate_feats = self.gate_features(x)
        gate_feats = torch.flatten(gate_feats, 1)
        
        # Compute gate logits
        gate_feats = F.gelu(self.ln1(self.fc1(gate_feats)))
        gate_feats = self.dropout(gate_feats)
        gate_logits = self.fc2(gate_feats)
        
        # Apply temperature scaling
        gate_logits = gate_logits / torch.clamp(self.temperature, min=0.5, max=5.0)
        
        # Add noise during training
        if self.training:
            noise = torch.randn_like(gate_logits) * torch.clamp(self.noise_scale, min=0.01, max=0.1)
            gate_logits = gate_logits + noise
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Normalize
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Create full gate weight tensor
        gates = torch.zeros_like(gate_logits)
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        return gates, top_k_indices, gate_logits


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.n_experts = 8
        self.top_k = 2
        
        in_channels = in_shape[1]
        self.output_dim = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape
        dropout = prm.get('dropout', 0.5)
        
        # Create 8 AlexNet experts
        self.experts = nn.ModuleList([
            AlexNetExpert(in_channels, self.output_dim, dropout, i)
            for i in range(self.n_experts)
        ])
        
        # Gating network
        self.gate = ImprovedGate(in_channels, self.n_experts, hidden_dim=128)
        
        # Diagnostic tracker
        self.utilization_tracker = ExpertUtilizationTracker(self.n_experts, self.output_dim)
        
        # Loss weights
        self.load_balance_weight = 0.005
        self.diversity_weight = 0.002
        
        # Label smoothing
        self.label_smoothing = 0.05
        
        self.to(device)
        self._print_memory_info()
    
    def _print_memory_info(self):
        param_count = sum(p.numel() for p in self.parameters())
        param_size_mb = param_count * 4 / (1024 * 1024)
        print(f"MoE-AlexNet Model with {self.n_experts} experts")
        print(f"Total parameters: {param_count:,}")
        print(f"Model size: {param_size_mb:.2f} MB")
    
    def forward(self, x):
        # Get gating decisions
        gate_weights, top_k_indices, gate_logits = self.gate(x)
        
        # Run all experts (can be optimized to run only top-k)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        
        # Weighted combination
        gate_weights_expanded = gate_weights.unsqueeze(1)
        final_output = torch.sum(expert_outputs * gate_weights_expanded, dim=2)
        
        # Store for auxiliary losses
        if self.training:
            self._last_gate_weights = gate_weights.detach()
            self._last_top_k_indices = top_k_indices.detach()
            
            # Store active expert outputs for diversity
            active_outputs = []
            unique_experts = torch.unique(top_k_indices)
            for expert_idx in unique_experts[:4]:
                active_outputs.append(expert_outputs[:, :, expert_idx].mean(dim=0).detach())
            self._last_active_outputs = active_outputs
        
        return final_output
    
    def compute_load_balance_loss(self, gate_weights, expert_indices):
        """Encourage balanced expert usage"""
        if gate_weights is None or gate_weights.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        
        expert_usage = gate_weights.sum(dim=0)
        target_usage = gate_weights.sum() / self.n_experts
        balance_loss = F.mse_loss(expert_usage, target_usage.expand_as(expert_usage))
        
        return torch.clamp(balance_loss, 0.0, 1.0)
    
    def compute_diversity_loss(self, expert_outputs):
        """Encourage diverse expert outputs"""
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
        
        # AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm.get('lr', 0.001),
            weight_decay=prm.get('weight_decay', 5e-4),
            betas=(0.9, 0.95),
            eps=1e-6
        )
        
        # Cosine annealing scheduler
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
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=3.0)
            
            self.optimizer.step()
            
            # Track accuracy
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
            
            # Cleanup
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