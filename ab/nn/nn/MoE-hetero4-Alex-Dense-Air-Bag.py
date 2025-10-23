import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Tuple
import torch.utils.checkpoint as cp


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


# ============================================================================
# ALEXNET EXPERT
# ============================================================================
class AlexNetExpert(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super().__init__()
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
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Linear(4096, num_classes)
        
    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.classifier(x)
        logits = self.final_layer(features)
        
        if return_features:
            return logits, features
        return logits


# ============================================================================
# DENSENET EXPERT
# ============================================================================
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        return self.conv1(self.relu1(self.norm1(concated_features)))

    def forward(self, input):
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input
        
        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNetExpert(nn.Module):
    def __init__(self, in_channels, num_classes, growth_rate=32, block_config=(6, 12, 24, 16), 
                 num_init_features=64, bn_size=4, drop_rate=0.0):
        super().__init__()
        
        self.features = nn.Sequential(
            OrderedDict([
                ("conv0", nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ("norm0", nn.BatchNorm2d(num_init_features)),
                ("relu0", nn.ReLU(inplace=True)),
                ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ])
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.final_features_dim = num_features
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        feature_vec = torch.flatten(out, 1)
        logits = self.classifier(feature_vec)
        
        if return_features:
            return logits, feature_vec
        return logits


# ============================================================================
# AIRNET EXPERT
# ============================================================================
class AirInitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class AirUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            ) if stride != 1 or in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.downsample(x)
        x = self.layers(x)
        return self.relu(x + residual)


class AirNetExpert(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        channels = [64, 128, 256, 512]
        init_block_channels = 64
        
        layers = [AirInitBlock(in_channels, init_block_channels)]
        for i, out_channels in enumerate(channels):
            layers.append(AirUnit(
                in_channels=init_block_channels if i == 0 else channels[i - 1],
                out_channels=out_channels,
                stride=1 if i == 0 else 2))
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.avgpool(x)
        feature_vec = torch.flatten(x, 1)
        logits = self.classifier(feature_vec)
        
        if return_features:
            return logits, feature_vec
        return logits


# ============================================================================
# BAGNET EXPERT
# ============================================================================
class BagNetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bottleneck_factor=4):
        super().__init__()
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = self.conv1x1_block(in_channels, mid_channels)
        self.conv2 = self.conv_block(mid_channels, mid_channels, kernel_size, stride)
        self.conv3 = self.conv1x1_block(mid_channels, out_channels, activation=False)

    def conv1x1_block(self, in_channels, out_channels, activation=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def conv_block(self, in_channels, out_channels, kernel_size, stride):
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class BagNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)
        self.body = BagNetBottleneck(in_channels, out_channels, kernel_size, stride)

        if self.resize_identity:
            self.identity_conv = self.conv1x1_block(in_channels, out_channels, activation=False)
        self.activ = nn.ReLU(inplace=True)

    def conv1x1_block(self, in_channels, out_channels, activation=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        if self.resize_identity:
            identity = self.identity_conv(x)

        x = self.body(x)

        if x.size(2) != identity.size(2) or x.size(3) != identity.size(3):
            identity = F.interpolate(identity, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return self.activ(x + identity)


class BagNetExpert(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        channels = [[64, 64, 64], [128, 128, 128], [256, 256, 256], [512, 512, 512]]
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        in_ch = 64
        for i, stage_channels in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(stage_channels):
                stride = 2 if (j == 0 and i > 0) else 1
                stage.add_module(f"unit{j + 1}", BagNetUnit(in_ch, out_channels, kernel_size=3, stride=stride))
                in_ch = out_channels
            self.features.add_module(f"stage{i + 1}", stage)

        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x, return_features=False):
        x = self.features(x)
        feature_vec = torch.flatten(x, 1)
        logits = self.classifier(feature_vec)
        
        if return_features:
            return logits, feature_vec
        return logits


# ============================================================================
# HETEROGENEOUS MOE GATE
# ============================================================================
class HeterogeneousGate(nn.Module):
    """Gating network that routes to 4 heterogeneous experts"""
    def __init__(self, input_channels, n_experts=4):
        super().__init__()
        self.n_experts = n_experts
        
        # Lightweight feature extractor for routing decision
        self.gate_features = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_experts)
        )
        
        # Temperature for controlling routing sharpness
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)
        
    def forward(self, x):
        # Extract routing features
        features = self.gate_features(x)
        features = features.flatten(1)
        
        # Compute gate logits
        gate_logits = self.gate(features)
        gate_logits = gate_logits / torch.clamp(self.temperature, 0.5, 5.0)
        
        # Training noise for exploration
        if self.training:
            noise = torch.randn_like(gate_logits) * 0.1
            gate_logits = gate_logits + noise
        
        # Softmax to get routing weights
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        return gate_weights, gate_logits


# ============================================================================
# COMPREHENSIVE DIAGNOSTICS
# ============================================================================
class ComprehensiveDiagnostics:
    """Advanced diagnostic system for heterogeneous MoE"""
    def __init__(self, expert_names, n_classes=10):
        self.expert_names = expert_names
        self.n_experts = len(expert_names)
        self.n_classes = n_classes
        self.reset()
        
    def reset(self):
        self.expert_usage = torch.zeros(self.n_experts)
        self.expert_confidence = torch.zeros(self.n_experts)
        self.expert_correct = torch.zeros(self.n_experts)
        self.expert_total = torch.zeros(self.n_experts)
        self.expert_class_predictions = torch.zeros(self.n_experts, self.n_classes)
        self.expert_class_correct = torch.zeros(self.n_experts, self.n_classes)
        self.routing_entropy_history = []
        self.batch_count = 0
        
    def update(self, gate_weights, predictions, targets, outputs):
        self.batch_count += 1
        batch_size = gate_weights.size(0)
        
        self.expert_usage += gate_weights.sum(dim=0).cpu()
        
        for batch_idx in range(batch_size):
            pred = predictions[batch_idx]
            target = targets[batch_idx]
            
            for expert_idx in range(self.n_experts):
                weight = gate_weights[batch_idx, expert_idx].item()
                if weight > 0.01:
                    self.expert_total[expert_idx] += weight
                    self.expert_confidence[expert_idx] += weight * outputs[batch_idx, pred].item()
                    self.expert_class_predictions[expert_idx, pred] += weight
                    
                    if pred == target:
                        self.expert_correct[expert_idx] += weight
                        self.expert_class_correct[expert_idx, target] += weight
        
        gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=1).mean()
        self.routing_entropy_history.append(gate_entropy.item())
    
    def print_report(self, epoch):
        usage_dist = self.expert_usage / (self.expert_usage.sum() + 1e-8)
        expert_acc = (self.expert_correct / (self.expert_total + 1e-8)).numpy()
        
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch} HETEROGENEOUS MOE DIAGNOSTICS")
        print(f"{'='*70}")
        
        print(f"\n📊 Expert Usage Distribution:")
        for i, name in enumerate(self.expert_names):
            print(f"   {name:12s}: {usage_dist[i]:.4f} ({usage_dist[i]*100:.1f}%)")
        
        print(f"\n🎯 Expert Performance:")
        for i, (name, acc) in enumerate(zip(self.expert_names, expert_acc)):
            print(f"   {name:12s}: Accuracy={acc:.4f} ({acc*100:.1f}%)")
        
        print(f"\n🔀 Routing Entropy: {np.mean(self.routing_entropy_history):.4f}")
        print(f"{'='*70}\n")


# ============================================================================
# HETEROGENEOUS MOE MODEL
# ============================================================================
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape
        self.dropout = prm.get('dropout', 0.5)
        
        # Create 4 heterogeneous experts
        print(f"\n{'='*70}")
        print("INITIALIZING HETEROGENEOUS MOE WITH 4 DIVERSE EXPERTS")
        print(f"{'='*70}")
        
        self.expert_names = ['AlexNet', 'DenseNet', 'AirNet', 'BagNet']
        
        self.experts = nn.ModuleList([
            AlexNetExpert(self.in_channels, self.num_classes, dropout=self.dropout),
            DenseNetExpert(self.in_channels, self.num_classes, drop_rate=self.dropout),
            AirNetExpert(self.in_channels, self.num_classes),
            BagNetExpert(self.in_channels, self.num_classes)
        ])
        
        # Gating network
        self.gate = HeterogeneousGate(self.in_channels, n_experts=4)
        
        # Loss weights
        self.load_balance_weight = 0.01
        self.label_smoothing = 0.1
        self.mixup_alpha = prm.get('mixup_alpha', 0.2)
        
        # Diagnostics
        self.diagnostics = ComprehensiveDiagnostics(self.expert_names, self.num_classes)
        
        self.to(device)
        self._print_info()
    
    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        expert_params = [sum(p.numel() for p in expert.parameters()) for expert in self.experts]
        gate_params = sum(p.numel() for p in self.gate.parameters())
        
        print(f"\nTotal Parameters: {total_params:,} ({total_params*4/(1024*1024):.2f} MB)")
        print(f"\nExpert Parameters:")
        for name, params in zip(self.expert_names, expert_params):
            print(f"  {name:12s}: {params:,} ({params*4/(1024*1024):.2f} MB)")
        print(f"  Gate Network: {gate_params:,} ({gate_params*4/(1024*1024):.2f} MB)")
        print(f"{'='*70}\n")
    
    def mixup_data(self, x, y, alpha=0.2):
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
        # Get gating weights
        gate_weights, gate_logits = self.gate(x)
        
        # Get outputs from all experts
        expert_outputs = []
        for expert in self.experts:
            output = expert(x)
            expert_outputs.append(output)
        
        # Stack expert outputs: [batch, num_classes, num_experts]
        expert_outputs = torch.stack(expert_outputs, dim=2)
        
        # Weighted combination: [batch, num_experts, 1]
        gate_weights_expanded = gate_weights.unsqueeze(1)
        
        # Final output: [batch, num_classes]
        final_output = torch.sum(expert_outputs * gate_weights_expanded, dim=2)
        
        # Store for diagnostics
        if self.training:
            self._last_gate_weights = gate_weights.detach()
            self._last_expert_outputs = expert_outputs.detach()
        
        return final_output
    
    def compute_load_balance_loss(self):
        if not hasattr(self, '_last_gate_weights'):
            return torch.tensor(0.0, device=self.device)
        
        gate_weights = self._last_gate_weights
        expert_usage = gate_weights.sum(dim=0)
        target_usage = gate_weights.sum() / 4  # 4 experts
        load_loss = F.mse_loss(expert_usage, target_usage.expand_as(expert_usage))
        
        return load_loss
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Different learning rates for different components
        gate_params = list(self.gate.parameters())
        expert_params = []
        for expert in self.experts:
            expert_params.extend(list(expert.parameters()))
        
        self.optimizer = torch.optim.AdamW([
            {'params': expert_params, 'lr': prm.get('lr', 0.001), 'weight_decay': prm.get('weight_decay', 5e-4)},
            {'params': gate_params, 'lr': prm.get('lr', 0.001) * 2, 'weight_decay': prm.get('weight_decay', 5e-4) / 2}
        ], betas=(0.9, 0.999))
        
        # Learning rate scheduler
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
                
                loss = lam * self.criteria(outputs, labels_a) + (1 - lam) * self.criteria(outputs, labels_b)
            else:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criteria(outputs, labels)
            
            # Add load balance loss
            load_loss = self.compute_load_balance_loss()
            total_loss_batch = loss + self.load_balance_weight * load_loss
            
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
        
        # Print results
        train_acc = 100. * correct / total
        print(f"\n📈 Training Accuracy: {train_acc:.2f}%")
        print(f"📉 Average Loss: {total_loss / num_batches:.4f}")
        print(f"🔧 Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        self.diagnostics.print_report(self.current_epoch)
        
        return total_loss / num_batches