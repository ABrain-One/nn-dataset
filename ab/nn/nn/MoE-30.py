import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


def supported_hyperparameters():
    return {'lr', 'momentum', 'weight_decay', 'dropout', 'load_balancing_loss_weight'}


class Expert(nn.Module):
    """Memory-efficient expert with optional dropout and layer norm"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)  # Remove bias to save memory
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights more efficiently
        self._init_weights()

    def _init_weights(self):
        # Use Xavier initialization for better convergence
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = F.gelu(x)  # GELU often works better than ReLU for transformers
        x = self.dropout(x)
        return self.fc2(x)


class GatingNetwork(nn.Module):
    """Improved gating network with load balancing"""
    def __init__(self, input_dim: int, num_experts: int, dropout: float = 0.1):
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize to encourage balanced expert usage
        nn.init.normal_(self.gate.weight, 0, 0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns gating weights and auxiliary loss for load balancing"""
        x = self.dropout(x)
        logits = self.gate(x)
        
        # Add noise for better load balancing during training
        if self.training:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise
            
        gating_weights = F.softmax(logits, dim=-1)
        
        # Compute load balancing loss (auxiliary loss)
        # Encourages equal usage of experts
        aux_loss = self._compute_load_balancing_loss(gating_weights)
        
        return gating_weights, aux_loss
    
    def _compute_load_balancing_loss(self, gating_weights: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary loss to encourage load balancing"""
        # Mean expert usage across batch
        expert_usage = gating_weights.mean(dim=(0, 1))  # [num_experts]
        
        # Coefficient of variation - penalize uneven usage
        mean_usage = expert_usage.mean()
        usage_var = ((expert_usage - mean_usage) ** 2).mean()
        cv = torch.sqrt(usage_var) / (mean_usage + 1e-8)
        
        return cv


class Net(nn.Module):
    def __init__(self, in_shape: Tuple, out_shape: Tuple, prm: Dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        
        # Configurable MoE parameters - make these smaller to reduce memory
        self.num_experts = prm.get('num_experts', 4)  # Reduced from 3 to 4 for better balance
        self.num_experts_per_tok = min(prm.get('num_experts_per_tok', 2), self.num_experts)
        self.dropout = prm.get('dropout', 0.1)
        self.load_balancing_weight = prm.get('load_balancing_loss_weight', 0.01)
        
        # Calculate dimensions more efficiently
        self.input_dim = self._calculate_input_dim(in_shape)
        self.output_dim = self._calculate_output_dim(out_shape)
        
        # Use adaptive hidden dimension based on input/output sizes
        self.hidden_dim = self._calculate_hidden_dim(prm)
        
        print(f"MoE Config - Experts: {self.num_experts}, Active: {self.num_experts_per_tok}")
        print(f"Dimensions - Input: {self.input_dim}, Hidden: {self.hidden_dim}, Output: {self.output_dim}")
        
        # Input projection (optional, for dimension reduction)
        self.input_projection = None
        if self.input_dim > 1024:  # Only project if input is very large
            projected_dim = min(512, self.input_dim // 2)
            self.input_projection = nn.Linear(self.input_dim, projected_dim, bias=False)
            self.input_dim = projected_dim
            print(f"Added input projection to {projected_dim} dimensions")

        # Create experts and gating network
        self.experts = nn.ModuleList([
            Expert(self.input_dim, self.hidden_dim, self.output_dim, self.dropout)
            for _ in range(self.num_experts)
        ])
        
        self.gate = GatingNetwork(self.input_dim, self.num_experts, self.dropout)
        
        # Output projection if needed
        self.output_projection = None
        if self.output_dim != out_shape[0]:
            self.output_projection = nn.Linear(self.output_dim, out_shape[0], bias=False)
        
        self.to(device)
        
        # Print memory usage estimate
        total_params = sum(p.numel() for p in self.parameters())
        memory_mb = total_params * 4 / 1024 / 1024  # Assuming float32
        print(f"Model parameters: {total_params:,}, Estimated memory: {memory_mb:.1f} MB")

    def _calculate_input_dim(self, in_shape: Tuple) -> int:
        """Calculate flattened input dimension"""
        if isinstance(in_shape, (list, tuple)) and len(in_shape) > 1:
            return math.prod(in_shape)
        return in_shape[0] if isinstance(in_shape, (list, tuple)) else in_shape

    def _calculate_output_dim(self, out_shape: Tuple) -> int:
        """Calculate output dimension"""
        return out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape

    def _calculate_hidden_dim(self, prm: Dict) -> int:
        """Calculate hidden dimension with memory constraints"""
        if 'hidden_dim' in prm:
            return prm['hidden_dim']
        
        # Adaptive sizing with memory constraints
        base_hidden = min(512, max(64, self.input_dim // 4, self.output_dim * 4))
        
        # Scale down for large models to prevent memory issues
        if self.input_dim > 10000:
            base_hidden = min(base_hidden, 256)
        
        return base_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Efficient input preprocessing
        x = x.float()
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        
        # Handle dimension mismatches efficiently
        if x.size(1) != self.input_dim:
            if self.input_projection is None:
                # Only pad/truncate if no projection layer
                if x.size(1) > self.input_dim:
                    x = x[:, :self.input_dim]
                else:
                    x = F.pad(x, (0, self.input_dim - x.size(1)))
        
        # Apply input projection if exists
        if self.input_projection is not None:
            x = self.input_projection(x)
        
        # Get gating scores and auxiliary loss
        gating_weights, aux_loss = self.gate(x)  # [batch, num_experts]
        
        # Top-k expert selection (more memory efficient)
        topk_weights, topk_indices = gating_weights.topk(
            self.num_experts_per_tok, dim=-1, sorted=False
        )
        
        # Renormalize selected weights
        topk_weights = F.normalize(topk_weights, p=1, dim=-1)
        
        # Memory-efficient expert computation
        # Instead of computing all experts, only compute selected ones
        output = torch.zeros(batch_size, self.output_dim, device=x.device, dtype=x.dtype)
        
        for i in range(self.num_experts_per_tok):
            expert_idx = topk_indices[:, i]  # [batch]
            expert_weights = topk_weights[:, i].unsqueeze(-1)  # [batch, 1]
            
            # Process each expert separately to save memory
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    expert_weight = expert_weights[mask]
                    output[mask] += expert_weight * expert_output
        
        # Apply output projection if exists
        if self.output_projection is not None:
            output = self.output_projection(output)
        
        # Store auxiliary loss for training
        self._aux_loss = aux_loss
        
        return output

    def get_auxiliary_loss(self) -> torch.Tensor:
        """Get the auxiliary loss for load balancing"""
        return getattr(self, '_aux_loss', torch.tensor(0.0, device=self.device))

    def train_setup(self, prm: Dict):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing
        
        # Use AdamW for better convergence
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm.get('lr', 1e-3),
            weight_decay=prm.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )
        
        # Optional: Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )

    def learn(self, train_data) -> float:
        self.train()
        total_loss = 0
        total_aux_loss = 0
        num_batches = 0

        for inputs, labels in train_data:
            inputs = inputs.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self(inputs)
            aux_loss = self.get_auxiliary_loss()

            # Handle output shapes
            if outputs.dim() > 2:
                outputs = outputs.view(outputs.size(0), -1)
            if labels.dim() > 1:
                labels = labels.view(-1)

            # Main loss + auxiliary loss
            main_loss = self.criteria(outputs, labels)
            total_loss_batch = main_loss + self.load_balancing_weight * aux_loss
            
            # Gradient clipping for stability
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            if hasattr(self, 'scheduler'):
                self.scheduler.step()

            total_loss += main_loss.item()
            total_aux_loss += aux_loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_aux_loss = total_aux_loss / max(num_batches, 1)
        
        if num_batches > 0 and num_batches % 10 == 0:  # Print every 10 batches
            print(f"Training - Main Loss: {avg_loss:.4f}, Aux Loss: {avg_aux_loss:.4f}")
        
        return avg_loss

    def evaluate(self, test_data) -> Tuple[float, float]:
        self.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_data:
                inputs = inputs.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                outputs = self(inputs)

                if outputs.dim() > 2:
                    outputs = outputs.view(outputs.size(0), -1)
                if labels.dim() > 1:
                    labels = labels.view(-1)

                loss = self.criteria(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / max(len(test_data), 1)
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy

    def get_expert_usage_stats(self) -> Dict:
        """Get statistics about expert usage for monitoring"""
        if not hasattr(self, '_aux_loss'):
            return {}
        
        # This would need to be implemented to track expert usage during training
        return {
            'num_experts': self.num_experts,
            'active_per_token': self.num_experts_per_tok,
            'aux_loss': self._aux_loss.item() if hasattr(self, '_aux_loss') else 0
        }


# Memory-efficient usage example
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration optimized for memory efficiency
    in_shape = (784,)
    out_shape = (10,)
    prm = {
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 0.1,
        'num_experts': 4,  # Reduced number of experts
        'num_experts_per_tok': 2,
        'hidden_dim': 256,  # Smaller hidden dimension
        'load_balancing_loss_weight': 0.01
    }

    # Create model
    model = Net(in_shape, out_shape, prm, device)
    model.train_setup(prm)

    print("\nMemory-efficient MoE model created!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with gradient checkpointing simulation
    with torch.no_grad():
        test_input = torch.randn(2, 784)  # Smaller batch size
        output = model(test_input)
        print(f"Test - Input: {test_input.shape}, Output: {output.shape}")
        
    # Memory usage tips
    print("\nMemory optimization tips:")
    print("1. Reduce batch_size in your training loop")
    print("2. Use gradient_checkpointing=True if available")
    print("3. Reduce num_experts or hidden_dim in prm")
    print("4. Use mixed precision training (torch.cuda.amp)")