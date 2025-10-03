# MoEv10-AlexNet: Enhanced MoE with AlexNet Experts (Type Error Fixed)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import defaultdict

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

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
        
        for batch_idx in range(batch_size):
            active_experts = top_k_indices[batch_idx]
            weights = gate_weights[batch_idx]
            
            for expert_idx in active_experts:
                if expert_idx < self.n_experts:
                    weight = weights[expert_idx].item()
                    self.expert_usage[expert_idx] += weight
                    self.expert_samples[expert_idx] += weight
                    
                    pred = predictions[batch_idx].argmax()
                    target = targets[batch_idx]
                    if pred == target:
                        self.expert_accuracy[expert_idx] += weight
                    
                    self.expert_class_confusion[expert_idx, target, pred] += weight
    
    def get_specialization_metrics(self):
        """Compute specialization metrics"""
        usage_dist = self.expert_usage / (self.expert_usage.sum() + 1e-8)
        usage_entropy = -(usage_dist * torch.log(usage_dist + 1e-8)).sum().item()
        
        expert_acc = (self.expert_accuracy / (self.expert_samples + 1e-8)).numpy()
        
        class_preferences = []
        for i in range(self.n_experts):
            if self.expert_samples[i] > 0:
                class_dist = self.expert_class_confusion[i].sum(dim=1)
                class_dist = class_dist / (class_dist.sum() + 1e-8)
                class_preferences.append(class_dist)
        
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
    """Enhanced AlexNet architecture with BatchNorm for better accuracy"""
    def __init__(self, in_channels, out_dim, dropout, expert_id):
        super(AlexNetExpert, self).__init__()
        self.expert_id = expert_id
        
        # Enhanced AlexNet with BatchNorm for stability
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Enhanced classifier with BatchNorm
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_dim),
        )
        
        # Better initialization
        self._init_weights(expert_id, out_dim)
    
    def _init_weights(self, expert_id, out_dim):
        """Better initialization strategy"""
        # Initialize conv layers
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Add slight class bias for specialization
        final_layer = self.classifier[-1]
        if final_layer.bias is not None:
            bias_values = torch.zeros(out_dim)
            preferred_classes = [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7],
                [8, 9],
            ]
            if expert_id < len(preferred_classes):
                classes = preferred_classes[expert_id]
                for cls in classes:
                    if cls < out_dim:
                        bias_values[cls] = 0.05
            final_layer.bias.data = bias_values
    
    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class EnhancedGate(nn.Module):
    """Improved gating network"""
    def __init__(self, input_channels, n_experts):
        super(EnhancedGate, self).__init__()
        self.n_experts = n_experts
        self.top_k = 2
        
        # Gating network
        self.gate_features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((3, 3))
        )
        
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, n_experts)
        
        self.dropout = nn.Dropout(0.2)
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        
        gate_feats = self.gate_features(x)
        gate_feats = torch.flatten(gate_feats, 1)
        
        gate_feats = self.fc1(gate_feats)
        gate_feats = self.bn1(gate_feats)
        gate_feats = F.relu(gate_feats)
        gate_feats = self.dropout(gate_feats)
        gate_logits = self.fc2(gate_feats)
        
        # Temperature scaling
        gate_logits = gate_logits / torch.clamp(self.temperature, min=0.1, max=3.0)
        
        # Add noise during training
        if self.training:
            noise = torch.randn_like(gate_logits) * 0.01
            gate_logits = gate_logits + noise
        
        # Select top-k
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Create sparse gate tensor
        gates = torch.zeros_like(gate_logits)
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        return gates, top_k_indices, gate_logits


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.n_experts = 4
        self.top_k = 2
        
        in_channels = in_shape[1]
        self.output_dim = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape
        dropout = prm.get('dropout', 0.5)
        
        # Create 4 enhanced AlexNet experts
        self.experts = nn.ModuleList([
            AlexNetExpert(in_channels, self.output_dim, dropout, i)
            for i in range(self.n_experts)
        ])
        
        # Enhanced gating
        self.gate = EnhancedGate(in_channels, self.n_experts)
        
        # Tracker
        self.utilization_tracker = ExpertUtilizationTracker(self.n_experts, self.output_dim)
        
        # Loss weights
        self.load_balance_weight = 0.0005
        self.diversity_weight = 0.0002
        
        self.best_train_acc = 0.0
        
        self.to(device)
        self._print_memory_info()
    
    def _print_memory_info(self):
        param_count = sum(p.numel() for p in self.parameters())
        param_size_mb = param_count * 4 / (1024 * 1024)
        print(f"Enhanced MoE-AlexNet with {self.n_experts} experts")
        print(f"Total parameters: {param_count:,}")
        print(f"Model size: {param_size_mb:.2f} MB")
    
    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        
        # Get gating decisions
        gate_weights, top_k_indices, gate_logits = self.gate(x)
        
        # Track active experts
        unique_experts = torch.unique(top_k_indices)
        expert_outputs_dict = {}
        
        # Compute outputs for active experts
        for expert_idx in unique_experts:
            expert_idx_val = expert_idx.item()
            expert_output = self.experts[expert_idx_val](x)
            expert_outputs_dict[expert_idx_val] = expert_output
        
        # Weighted combination
        batch_size = x.size(0)
        final_output = torch.zeros(batch_size, self.output_dim, device=x.device, dtype=torch.float32)
        
        for expert_idx_val, expert_output in expert_outputs_dict.items():
            expert_weights = gate_weights[:, expert_idx_val].unsqueeze(1)
            final_output = final_output + expert_weights * expert_output
        
        # Store for auxiliary losses
        if self.training:
            self._last_gate_weights = gate_weights.detach()
            self._last_top_k_indices = top_k_indices.detach()
            
            expert_outputs_for_diversity = []
            for expert_idx in list(expert_outputs_dict.keys())[:3]:
                expert_outputs_for_diversity.append(
                    expert_outputs_dict[expert_idx][:2].mean(dim=0).detach()
                )
            self._last_active_outputs = expert_outputs_for_diversity
        
        return final_output
    
    def compute_load_balance_loss(self, gate_weights):
        """Encourage balanced expert usage"""
        if gate_weights is None or gate_weights.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        
        expert_usage = gate_weights.sum(dim=0)
        mean_usage = expert_usage.mean()
        balance_loss = ((expert_usage - mean_usage) ** 2).mean()
        
        return balance_loss
    
    def compute_diversity_loss(self, expert_outputs):
        """Encourage diverse expert outputs"""
        if not expert_outputs or len(expert_outputs) < 2:
            return torch.tensor(0.0, device=self.device)
        
        similarities = []
        for i in range(len(expert_outputs)):
            for j in range(i + 1, len(expert_outputs)):
                sim = F.cosine_similarity(
                    expert_outputs[i].unsqueeze(0), 
                    expert_outputs[j].unsqueeze(0)
                )
                similarities.append(sim)
        
        if similarities:
            return torch.stack(similarities).mean()
        return torch.tensor(0.0, device=self.device)
    
    def train_setup(self, prm):
        self.to(self.device)
        
        # Standard CrossEntropyLoss
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        
        # SGD with Nesterov momentum
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            nesterov=True
        )
    
    def learn(self, train_data):
        self.train()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        self.utilization_tracker.reset()
        
        for inputs, labels in train_data:
            # Ensure correct types
            inputs = inputs.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)
            
            self.optimizer.zero_grad()
            
            outputs = self(inputs)
            main_loss = self.criteria[0](outputs, labels)
            
            # Auxiliary losses
            aux_loss = torch.tensor(0.0, device=self.device)
            
            if hasattr(self, '_last_gate_weights'):
                load_loss = self.compute_load_balance_loss(
                    self._last_gate_weights
                ) * self.load_balance_weight
                aux_loss = aux_loss + load_loss
            
            if hasattr(self, '_last_active_outputs'):
                div_loss = self.compute_diversity_loss(
                    self._last_active_outputs
                ) * self.diversity_weight
                aux_loss = aux_loss + div_loss
            
            total_loss_batch = main_loss + aux_loss
            total_loss_batch.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Track accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update tracker
            if hasattr(self, '_last_gate_weights'):
                self.utilization_tracker.update(
                    self._last_gate_weights, predicted, labels, self._last_top_k_indices
                )
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            # Cleanup
            for attr in ['_last_gate_weights', '_last_top_k_indices', '_last_active_outputs']:
                if hasattr(self, attr):
                    delattr(self, attr)
            
            # Clear cache periodically
            if num_batches % 25 == 0:
                torch.cuda.empty_cache()
        
        # Metrics
        train_acc = 100. * correct / total
        metrics = self.utilization_tracker.get_specialization_metrics()
        
        if train_acc > self.best_train_acc:
            self.best_train_acc = train_acc
        
        print(f"Training Accuracy: {train_acc:.2f}% (Best: {self.best_train_acc:.2f}%)")
        print(f"Expert Usage: {metrics['usage_distribution']}")
        print(f"Expert Diversity: {metrics['average_diversity']:.3f}")
        
        return total_loss / num_batches if num_batches > 0 else 0