# MoE model that follows AlexNet pattern

import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class SimpleExpert(nn.Module):
    """Simple feedforward expert network"""
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class SimpleGate(nn.Module):
    """Simple gating network"""
    def __init__(self, input_dim, n_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_experts)
        )
        self.n_experts = n_experts
        self.top_k = 2
    
    def forward(self, x):
        gate_logits = self.gate(x)
        
        # Top-k gating
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Create sparse gate weights
        gates = torch.zeros_like(gate_logits)
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        return gates


class Net(nn.Module):
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.n_experts = 6
        
        # Same CNN features as AlexNet
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=11, stride=4, padding=2),
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
        
        dropout: float = prm['dropout']
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Feature dimension after CNN + pooling
        feature_dim = 256 * 6 * 6  # 9216
        
        # Create 6 simple experts
        self.experts = nn.ModuleList([
            SimpleExpert(feature_dim, out_shape[0], dropout) 
            for _ in range(self.n_experts)
        ])
        
        # Gating network
        self.gate = SimpleGate(feature_dim, self.n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features using AlexNet backbone
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Shape: [batch_size, 9216]
        
        # Get gating weights
        gate_weights = self.gate(x)  # Shape: [batch_size, n_experts]
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        # Stack expert outputs: [batch_size, output_dim, n_experts]
        expert_outputs = torch.stack(expert_outputs, dim=2)
        
        # Apply gating weights
        gate_weights = gate_weights.unsqueeze(1)  # [batch_size, 1, n_experts]
        final_output = torch.sum(expert_outputs * gate_weights, dim=2)
        
        return final_output