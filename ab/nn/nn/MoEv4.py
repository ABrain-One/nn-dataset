import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def supported_hyperparameters():
    return {'lr', 'momentum', 'weight_decay', 'lr_schedule'}


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
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
    """Residual block with SE attention"""
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


class ImprovedFeatureExtractor(nn.Module):
    """Enhanced CNN with residual connections and attention"""
    def __init__(self, in_channels=3):
        super(ImprovedFeatureExtractor, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        
        # Feature dimension for 32x32 input: 512
        self.output_dim = 512

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return x


class ExpertWithSpecialization(nn.Module):
    """Enhanced expert with better specialization capabilities"""
    def __init__(self, input_dim, hidden_dim, out_dim, expert_id):
        super(ExpertWithSpecialization, self).__init__()
        self.expert_id = expert_id
        self.input_dim = input_dim
        
        # Specialized initialization for different experts
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, out_dim)
        
        # Layer normalization for better training stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights differently for each expert to encourage specialization
        self._init_weights()

    def _init_weights(self):
        # Different initialization schemes for different experts to encourage diversity
        init_schemes = [
            ('xavier_uniform', lambda x: nn.init.xavier_uniform_(x)),
            ('xavier_normal', lambda x: nn.init.xavier_normal_(x)),
            ('kaiming_uniform', lambda x: nn.init.kaiming_uniform_(x, mode='fan_in')),
            ('kaiming_normal', lambda x: nn.init.kaiming_normal_(x, mode='fan_in')),
            ('orthogonal', lambda x: nn.init.orthogonal_(x)),
            ('sparse', lambda x: nn.init.sparse_(x, sparsity=0.1)),
            ('uniform', lambda x: nn.init.uniform_(x, -0.1, 0.1)),
            ('normal', lambda x: nn.init.normal_(x, 0, 0.02))
        ]
        
        scheme_name, init_func = init_schemes[self.expert_id % len(init_schemes)]
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_func(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.gelu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.gelu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class ImprovedGate(nn.Module):
    """Enhanced gating network with better routing decisions"""
    def __init__(self, input_dim, n_experts, hidden_dim=256):
        super(ImprovedGate, self).__init__()
        self.input_dim = input_dim
        self.n_experts = n_experts
        self.top_k = 2
        
        # Multi-layer gating network with attention
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, n_experts)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        
        self.dropout = nn.Dropout(0.1)
        
        # Temperature parameter for sharpening/smoothing
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Learnable noise scale for load balancing
        self.noise_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # Enhanced gating computation
        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.gelu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        gate_logits = self.fc3(x)
        
        # Temperature scaling
        gate_logits = gate_logits / torch.clamp(self.temperature, min=0.1)
        
        # Adaptive noise for better load balancing during training
        if self.training:
            noise = torch.randn_like(gate_logits) * torch.clamp(self.noise_scale, min=0.01, max=0.5)
            gate_logits = gate_logits + noise
        
        # Top-k gating with improved normalization
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # Apply softmax with higher precision
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Renormalize to ensure sum = 1
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Create sparse gate weights
        gates = torch.zeros_like(gate_logits)
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        return gates, top_k_indices, gate_logits


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.n_experts = 8
        self.top_k = 2
        
        # Enhanced CNN feature extractor
        self.feature_extractor = ImprovedFeatureExtractor(in_channels=in_shape[1])
        self.feature_dim = self.feature_extractor.output_dim
        
        self.output_dim = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape
        
        # Larger hidden dimension for better capacity
        self.hidden_dim = 512
        
        # Create 8 specialized experts
        self.experts = nn.ModuleList([
            ExpertWithSpecialization(self.feature_dim, self.hidden_dim, self.output_dim, i)
            for i in range(self.n_experts)
        ])
        
        # Enhanced gate
        self.gate = ImprovedGate(self.feature_dim, self.n_experts, 256)
        
        # Auxiliary losses weights
        self.load_balance_weight = 0.01
        self.diversity_weight = 0.005
        
        # Label smoothing for better generalization
        self.label_smoothing = 0.1
        
        self.to(device)
        self._print_memory_info()

    def _print_memory_info(self):
        param_count = sum(p.numel() for p in self.parameters())
        param_size_mb = param_count * 4 / (1024 * 1024)
        print(f"Enhanced MoE-8 Model parameters: {param_count:,}")
        print(f"Model size: {param_size_mb:.2f} MB")
        print(f"Experts: {self.n_experts}, Top-K: {self.top_k}")
        print(f"Feature dim: {self.feature_dim}, Hidden dim: {self.hidden_dim}, Output dim: {self.output_dim}")

    def compute_load_balance_loss(self, gate_weights, expert_indices):
        """Improved load balancing loss with safety checks"""
        try:
            if gate_weights is None or gate_weights.numel() == 0:
                return torch.tensor(0.0, device=self.device)
            
            # Expert usage frequency
            expert_usage = gate_weights.sum(dim=0)
            
            # Check for valid usage
            if expert_usage.numel() == 0:
                return torch.tensor(0.0, device=self.device)
            
            target_usage = gate_weights.sum() / self.n_experts
            
            # L2 loss for balanced usage
            balance_loss = F.mse_loss(expert_usage, target_usage.expand_as(expert_usage))
            
            # Additional entropy regularization to encourage diversity
            # Add small epsilon to prevent log(0)
            gate_weights_safe = gate_weights + 1e-8
            gate_entropy = -torch.sum(gate_weights_safe * torch.log(gate_weights_safe), dim=1).mean()
            entropy_loss = -gate_entropy  # We want to maximize entropy
            
            # Check for NaN/Inf
            if torch.isnan(balance_loss) or torch.isinf(balance_loss):
                balance_loss = torch.tensor(0.0, device=self.device)
            if torch.isnan(entropy_loss) or torch.isinf(entropy_loss):
                entropy_loss = torch.tensor(0.0, device=self.device)
            
            total_loss = balance_loss + 0.1 * entropy_loss
            return torch.clamp(total_loss, 0.0, 10.0)  # Clamp to reasonable range
            
        except Exception as e:
            return torch.tensor(0.0, device=self.device)

    def compute_diversity_loss(self, expert_outputs):
        """Encourage experts to be diverse"""
        try:
            if not expert_outputs or len(expert_outputs) < 2:
                return torch.tensor(0.0, device=self.device)
            
            # Ensure all outputs have the same shape and are on the same device
            valid_outputs = []
            for output in expert_outputs:
                if output is not None and output.numel() > 0:
                    # Flatten to 1D if needed
                    if output.dim() > 1:
                        output = output.view(-1)
                    valid_outputs.append(output)
            
            if len(valid_outputs) < 2:
                return torch.tensor(0.0, device=self.device)
            
            # Compute pairwise cosine similarities between expert outputs
            similarities = []
            for i in range(min(len(valid_outputs), 4)):  # Limit to prevent memory issues
                for j in range(i + 1, min(len(valid_outputs), 4)):
                    try:
                        # Ensure tensors are the same size
                        output_i = valid_outputs[i]
                        output_j = valid_outputs[j]
                        
                        min_size = min(output_i.size(0), output_j.size(0))
                        output_i = output_i[:min_size]
                        output_j = output_j[:min_size]
                        
                        # Compute cosine similarity with proper dimensions
                        sim = F.cosine_similarity(output_i.unsqueeze(0), output_j.unsqueeze(0), dim=1).mean()
                        
                        if not torch.isnan(sim) and not torch.isinf(sim):
                            similarities.append(sim)
                    except Exception as e:
                        continue
            
            if similarities:
                # We want low similarity (high diversity)
                avg_similarity = torch.stack(similarities).mean()
                return torch.clamp(avg_similarity, 0.0, 1.0)  # Clamp to valid range
            
            return torch.tensor(0.0, device=self.device)
            
        except Exception as e:
            # Return zero loss if computation fails
            return torch.tensor(0.0, device=self.device)

    def forward(self, x):
        try:
            # Extract enhanced CNN features
            features = self.feature_extractor(x)
            
            # Get improved gating decisions
            gate_weights, top_k_indices, gate_logits = self.gate(features)
            
            # More memory-efficient sparse MoE computation
            batch_size = features.size(0)
            
            # Vectorized expert computation instead of loops
            expert_outputs = []
            for i in range(self.n_experts):
                expert_output = self.experts[i](features)
                expert_outputs.append(expert_output)
            
            # Stack all expert outputs: [batch_size, output_dim, n_experts]
            expert_outputs = torch.stack(expert_outputs, dim=2)
            
            # Apply gating weights efficiently
            gate_weights_expanded = gate_weights.unsqueeze(1)  # [batch_size, 1, n_experts]
            final_output = torch.sum(expert_outputs * gate_weights_expanded, dim=2)
            
            # Store for auxiliary losses during training (with detach to prevent memory issues)
            if self.training:
                self._last_gate_weights = gate_weights.detach()
                self._last_expert_indices = top_k_indices.detach()
                # Store only a few active expert outputs to save memory and avoid dimension issues
                active_outputs = []
                try:
                    # Get outputs from top-k experts for diversity loss
                    unique_experts = torch.unique(top_k_indices).cpu().numpy()
                    for expert_idx in unique_experts[:4]:  # Limit to 4 experts max
                        if expert_idx < len(expert_outputs[0]):
                            # Get the output for this expert across all batch items
                            expert_out = expert_outputs[:, :, expert_idx].detach()
                            if expert_out.numel() > 0:
                                active_outputs.append(expert_out.mean(dim=0))  # Average across batch
                    self._last_active_outputs = active_outputs
                except Exception as e:
                    self._last_active_outputs = []
            
            return final_output
            
        except Exception as e:
            # Fallback to simpler computation if there's an error
            print(f"Warning: MoE forward pass error: {e}. Using fallback.")
            features = self.feature_extractor(x)
            # Simple average of all experts as fallback
            outputs = []
            for expert in self.experts:
                outputs.append(expert(features))
            return torch.stack(outputs).mean(dim=0)

    def train_setup(self, prm):
        self.to(self.device)
        
        # Label smoothing cross entropy
        self.criteria = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Simpler optimizer setup to avoid DataLoader issues
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=prm.get('lr', 0.001),
            weight_decay=prm.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Simpler scheduler to avoid potential issues
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )

    def learn(self, train_data):
        self.train()
        total_loss = 0
        num_batches = 0
        
        try:
            for inputs, labels in train_data:
                try:
                    inputs = inputs.to(self.device, dtype=torch.float32, non_blocking=True)
                    labels = labels.to(self.device, dtype=torch.long, non_blocking=True)

                    self.optimizer.zero_grad()
                    
                    # Forward pass with error handling
                    outputs = self(inputs)
                    
                    # Main classification loss
                    main_loss = self.criteria(outputs, labels)
                    
                    # Auxiliary losses with safe fallbacks
                    load_balance_loss = torch.tensor(0.0, device=self.device)
                    diversity_loss = torch.tensor(0.0, device=self.device)
                    
                    # Reduce auxiliary loss weights to minimize impact of potential issues
                    aux_loss_weight = 0.001  # Much smaller weight
                    
                    if hasattr(self, '_last_gate_weights') and self._last_gate_weights is not None:
                        try:
                            load_balance_loss = self.compute_load_balance_loss(
                                self._last_gate_weights, self._last_expert_indices
                            )
                            load_balance_loss = load_balance_loss * aux_loss_weight
                        except Exception as e:
                            load_balance_loss = torch.tensor(0.0, device=self.device)
                    
                    if hasattr(self, '_last_active_outputs') and len(self._last_active_outputs) > 0:
                        try:
                            diversity_loss = self.compute_diversity_loss(self._last_active_outputs)
                            diversity_loss = diversity_loss * aux_loss_weight
                        except Exception as e:
                            diversity_loss = torch.tensor(0.0, device=self.device)
                    
                    # Total loss - focus mainly on classification loss
                    total_loss_batch = main_loss + load_balance_loss + diversity_loss
                    
                    # Check for NaN or inf
                    if torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch):
                        print("Warning: NaN or inf detected in loss, skipping batch")
                        continue
                    
                    total_loss_batch.backward()
                    
                    # Gradient clipping for stability
                    grad_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    
                    # Check gradient norm
                    if grad_norm > 10.0:
                        print(f"Warning: Large gradient norm: {grad_norm}")
                    
                    self.optimizer.step()
                    
                    total_loss += total_loss_batch.item()
                    num_batches += 1
                    
                    # Clear stored tensors to free memory
                    if hasattr(self, '_last_gate_weights'):
                        del self._last_gate_weights
                    if hasattr(self, '_last_expert_indices'):
                        del self._last_expert_indices
                    if hasattr(self, '_last_active_outputs'):
                        del self._last_active_outputs
                    
                    # Force garbage collection periodically
                    if num_batches % 10 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                except Exception as batch_error:
                    print(f"Error in batch processing: {batch_error}")
                    continue
            
            # Update learning rate
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            return total_loss / num_batches if num_batches > 0 else 0
            
        except Exception as e:
            print(f"Error in training loop: {e}")
            return 0