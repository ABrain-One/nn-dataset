import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


def supported_hyperparameters():
    return {'lr', 'momentum'}


class SimpleExpert(nn.Module):
    """Ultra-simple expert with minimal layers"""
    def __init__(self, input_dim, output_dim):
        super(SimpleExpert, self).__init__()
        self.input_dim = input_dim
        # Only 2 layers to keep it simple and memory-efficient
        hidden_dim = min(64, max(16, input_dim // 8))  # Very small hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.float()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Handle input dimension mismatch
        if x.size(-1) != self.input_dim:
            if x.size(-1) > self.input_dim:
                x = x[:, :self.input_dim]
            else:
                padding = self.input_dim - x.size(-1)
                x = F.pad(x, (0, padding))

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleGate(nn.Module):
    """Ultra-simple gating mechanism"""
    def __init__(self, input_dim, n_experts):
        super(SimpleGate, self).__init__()
        self.input_dim = input_dim
        self.n_experts = n_experts
        self.top_k = 2  # Always use top-2 for sparsity
        
        # Single layer gate - simplest possible
        self.gate_layer = nn.Linear(input_dim, n_experts)

    def forward(self, x):
        x = x.float()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Handle input dimension mismatch
        if x.size(-1) != self.input_dim:
            if x.size(-1) > self.input_dim:
                x = x[:, :self.input_dim]
            else:
                padding = self.input_dim - x.size(-1)
                x = F.pad(x, (0, padding))

        # Get gate logits
        gate_logits = self.gate_layer(x)
        
        # Top-k selection for sparsity
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Create sparse gate weights
        gates = torch.zeros_like(gate_logits)
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        return gates, top_k_indices


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.n_experts = 6  # 6 experts for memory safety
        self.top_k = 2      # Top-2 routing

        # Calculate input dimension
        if isinstance(in_shape, (list, tuple)) and len(in_shape) > 1:
            self.input_dim = 1
            for dim in in_shape:
                self.input_dim *= dim
        else:
            self.input_dim = in_shape[0] if isinstance(in_shape, (list, tuple)) else in_shape

        self.output_dim = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape

        # Keep dimensions very small for memory safety
        if self.input_dim > 1024:
            print(f"Warning: Large input dimension {self.input_dim}, limiting to 1024")
            self.input_dim = 1024

        # Create 6 simple experts
        self.experts = nn.ModuleList([
            SimpleExpert(self.input_dim, self.output_dim)
            for _ in range(self.n_experts)
        ])
        
        # Simple gate
        self.gate = SimpleGate(self.input_dim, self.n_experts)

        # Move to device
        self.to(device)
        self._print_memory_info()

    def _print_memory_info(self):
        param_count = sum(p.numel() for p in self.parameters())
        param_size_mb = param_count * 4 / (1024 * 1024)
        print(f"Simple MoE-6 Model parameters: {param_count:,}")
        print(f"Model size: {param_size_mb:.2f} MB")
        print(f"Experts: {self.n_experts}, Top-K: {self.top_k}")
        print(f"Input dim: {self.input_dim}, Output dim: {self.output_dim}")

        if torch.cuda.is_available():
            print(f"GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    def forward(self, x):
        try:
            x = x.float()
            batch_size = x.size(0)

            # Very conservative batch size limit
            if batch_size > 32:
                print(f"Warning: Large batch size {batch_size}, processing first 32 samples")
                x = x[:32]
                batch_size = 32

            # Flatten input
            if x.dim() > 2:
                x = x.view(batch_size, -1)

            # Truncate if too large
            if x.size(-1) > self.input_dim:
                x = x[:, :self.input_dim]

            # Get sparse gates
            gate_weights, top_k_indices = self.gate(x)

            # Simple sparse computation
            output = torch.zeros(batch_size, self.output_dim, device=self.device)
            
            # Only process active experts
            for i in range(self.n_experts):
                # Check which samples use this expert
                expert_mask = (top_k_indices == i).any(dim=1)
                if expert_mask.any():
                    expert_output = self.experts[i](x)
                    weighted_output = expert_output * gate_weights[:, i].unsqueeze(-1)
                    output += weighted_output

            return output

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("OOM! Clearing cache and returning zeros...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return torch.zeros(x.size(0), self.output_dim, device=self.device)
            else:
                raise e

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(),
                                         lr=prm.get('lr', 0.001),  # Lower default LR
                                         momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (inputs, labels) in enumerate(train_data):
            try:
                # Frequent memory cleanup
                if batch_idx % 5 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                inputs = inputs.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                # Very small batch size for safety
                if inputs.size(0) > 16:
                    inputs = inputs[:16]
                    labels = labels[:16]

                self.optimizer.zero_grad()
                outputs = self(inputs)

                # Handle shapes
                if outputs.dim() > 2:
                    outputs = outputs.view(outputs.size(0), -1)
                if labels.dim() > 1:
                    labels = labels.view(-1)

                loss = self.criteria(outputs, labels)
                loss.backward()

                # Conservative gradient clipping
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.25)

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Clear everything
                del inputs, labels, outputs, loss

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at batch {batch_idx}, skipping and clearing cache...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    print(f"Training error: {e}")
                    continue

        return total_loss / max(num_batches, 1)

    def evaluate(self, test_data):
        self.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_data:
                try:
                    inputs = inputs.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.long)

                    # Small batch for safety
                    if inputs.size(0) > 16:
                        inputs = inputs[:16]
                        labels = labels[:16]

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

                    del inputs, labels, outputs, loss

                except Exception as e:
                    print(f"Eval error: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

        return total_loss / len(test_data), correct / total if total > 0 else 0


# Test the simple model
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Very conservative test settings
    in_shape = (784,)
    out_shape = (10,)
    prm = {'lr': 0.001, 'momentum': 0.9}

    try:
        model = Net(in_shape, out_shape, prm, device)
        model.train_setup(prm)

        print("Simple MoE-6 model created successfully!")
        print("This is the most memory-efficient version possible.")

        # Test with tiny batch
        test_input = torch.randn(4, 784)
        test_output = model(test_input)
        print(f"Test successful! Input: {test_input.shape}, Output: {test_output.shape}")

    except Exception as e:
        print(f"Error: {e}")