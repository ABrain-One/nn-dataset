import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(Expert, self).__init__()
        self.input_dim = input_dim
        # More capacity for better performance
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Gate(nn.Module):
    def __init__(self, input_dim, n_experts, hidden_dim=64):
        super(Gate, self).__init__()
        self.input_dim = input_dim
        self.n_experts = n_experts
        self.top_k = 2
        
        # Better gating network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_experts)
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        x = self.dropout(x)
        gate_logits = self.fc2(x)
        
        # Add noise for load balancing during training
        if self.training:
            noise = torch.randn_like(gate_logits) * 0.01
            gate_logits = gate_logits + noise
        
        # Top-k gating
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Create sparse gate weights
        gates = torch.zeros_like(gate_logits)
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        return gates, top_k_indices


class ConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(ConvFeatureExtractor, self).__init__()
        
        # Efficient CNN backbone for CIFAR-10
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)
        
        # Calculate output dimension: 128 * 4 * 4 = 2048 for CIFAR-10
        self.output_dim = 128 * 4 * 4
        
    def forward(self, x):
        # Block 1: 32x32 -> 16x16
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Block 2: 16x16 -> 8x8
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Block 3: 8x8 -> 4x4
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.n_experts = 8
        self.top_k = 2
        
        # CNN feature extractor
        self.feature_extractor = ConvFeatureExtractor(in_channels=in_shape[1])
        self.feature_dim = self.feature_extractor.output_dim
        
        self.output_dim = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape
        
        # Expert hidden dimension - balanced for performance vs memory
        self.hidden_dim = 256
        
        # Create 8 experts
        self.experts = nn.ModuleList([
            Expert(self.feature_dim, self.hidden_dim, self.output_dim)
            for _ in range(self.n_experts)
        ])
        
        # Gate for top-2 routing
        self.gate = Gate(self.feature_dim, self.n_experts, 128)
        
        # Load balancing
        self.load_balance_weight = 0.01
        
        self.to(device)
        self._print_memory_info()

    def _print_memory_info(self):
        param_count = sum(p.numel() for p in self.parameters())
        param_size_mb = param_count * 4 / (1024 * 1024)
        print(f"MoE-8 Model parameters: {param_count:,}")
        print(f"Model size: {param_size_mb:.2f} MB")
        print(f"Experts: {self.n_experts}, Top-K: {self.top_k}")
        print(f"Feature dim: {self.feature_dim}, Hidden dim: {self.hidden_dim}, Output dim: {self.output_dim}")

    def compute_load_balance_loss(self, gate_weights):
        # Encourage balanced expert usage
        expert_usage = gate_weights.sum(dim=0)
        target_usage = gate_weights.sum() / self.n_experts
        load_balance_loss = F.mse_loss(expert_usage, target_usage.expand_as(expert_usage))
        return load_balance_loss

    def forward(self, x):
        # Extract CNN features
        features = self.feature_extractor(x)
        
        # Get gating decisions
        gate_weights, top_k_indices = self.gate(features)
        
        # Sparse MoE computation - more efficient implementation
        outputs = []
        for i in range(self.n_experts):
            expert_output = self.experts[i](features)
            outputs.append(expert_output)
        
        # Stack all expert outputs
        expert_outputs = torch.stack(outputs, dim=2)  # [batch, output_dim, n_experts]
        
        # Apply gating weights
        gate_weights_expanded = gate_weights.unsqueeze(1)  # [batch, 1, n_experts]
        final_output = (expert_outputs * gate_weights_expanded).sum(dim=2)
        
        return final_output

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(),
                                         lr=prm.get('lr', 0.01),
                                         momentum=prm.get('momentum', 0.9),
                                         weight_decay=1e-4)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self(inputs)
            
            # Main classification loss
            main_loss = self.criteria(outputs, labels)
            
            # Load balancing loss
            features = self.feature_extractor(inputs)
            gate_weights, _ = self.gate(features)
            load_balance_loss = self.compute_load_balance_loss(gate_weights)
            
            # Total loss
            total_loss = main_loss + self.load_balance_weight * load_balance_loss
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()