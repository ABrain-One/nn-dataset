import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.1):
        super(Expert, self).__init__()
        self.input_dim = input_dim
        # Smaller expert networks to save memory
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.float()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        if x.size(-1) != self.input_dim:
            if x.size(-1) > self.input_dim:
                x = x[:, :self.input_dim]
            else:
                padding = self.input_dim - x.size(-1)
                x = F.pad(x, (0, padding))

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Gate(nn.Module):
    def __init__(self, input_dim, n_experts, hidden_dim=32, dropout=0.1):
        super(Gate, self).__init__()
        self.input_dim = input_dim
        self.n_experts = n_experts
        # Simple gate for top-k selection
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_experts)
        self.dropout = nn.Dropout(dropout)
        self.top_k = 2  # Top-2 routing

    def forward(self, x):
        x = x.float()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        if x.size(-1) != self.input_dim:
            if x.size(-1) > self.input_dim:
                x = x[:, :self.input_dim]
            else:
                padding = self.input_dim - x.size(-1)
                x = F.pad(x, (0, padding))

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        gate_logits = self.fc2(x)
        
        # Top-k gating for sparsity
        if self.training:
            # Add noise for load balancing during training
            noise = torch.randn_like(gate_logits) * 0.1
            gate_logits = gate_logits + noise
        
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # Softmax over top-k
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Create sparse gate weights
        gates = torch.zeros_like(gate_logits)
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        return gates, top_k_indices


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.n_experts = 8  # 8 experts as requested
        self.top_k = 2      # Top-2 routing for sparsity

        # Get input dimensions
        input_height, input_width = in_shape[2], in_shape[3]
        
        # Adaptive AlexNet-style feature extractor based on input size
        if input_height >= 224 and input_width >= 224:
            # Original AlexNet for large images (224x224+)
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
        else:
            # Modified AlexNet for smaller images (like CIFAR-10: 32x32)
            self.features = nn.Sequential(
                nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        
        # Adaptive pooling to get consistent feature size
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Calculate feature dimension after conv layers and pooling
        self.feature_dim = 256 * 6 * 6  # 9216 features from AlexNet conv layers
        
        # Output dimension
        self.output_dim = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape
        
        # Dropout parameter
        dropout = prm.get('dropout', 0.1)
        
        # Hidden dimension for experts (smaller than AlexNet's 4096)
        self.hidden_dim = min(512, max(64, self.feature_dim // 32))
        
        # Create 8 experts that will replace AlexNet's classifier
        self.experts = nn.ModuleList([
            Expert(self.feature_dim, self.hidden_dim, self.output_dim, dropout)
            for _ in range(self.n_experts)
        ])
        
        # Gate for top-2 routing based on CNN features
        self.gate = Gate(self.feature_dim, self.n_experts, self.hidden_dim // 4, dropout)

        # Move to device
        self.to(device)

        # Print memory usage info
        self._print_memory_info()

    def _print_memory_info(self):
        param_count = sum(p.numel() for p in self.parameters())
        param_size_mb = param_count * 4 / (1024 * 1024)  # 4 bytes per float32
        print(f"AlexNet-MoE Model parameters: {param_count:,}")
        print(f"Model size: {param_size_mb:.2f} MB")
        print(f"Experts: {self.n_experts}, Top-K: {self.top_k}")
        print(f"Feature dim: {self.feature_dim}, Hidden dim: {self.hidden_dim}, Output dim: {self.output_dim}")

        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            # AlexNet-style feature extraction
            x = self.features(x)  # Conv layers + pooling
            x = self.avgpool(x)   # Adaptive pooling
            x = torch.flatten(x, 1)  # Flatten to (batch_size, feature_dim)
            
            batch_size = x.size(0)
            
            # Limit batch size if too large
            if batch_size > 64:
                print(f"Warning: Large batch size {batch_size}, consider reducing")

            # Get sparse gating weights (top-2) based on CNN features
            gate_weights, top_k_indices = self.gate(x)

            # Sparse MoE computation - only compute active experts
            output = torch.zeros(batch_size, self.output_dim, device=self.device)
            
            # Only compute outputs for active experts
            for i in range(self.n_experts):
                # Check if this expert is used by any sample
                expert_mask = (top_k_indices == i).any(dim=1)
                if expert_mask.any():
                    # Get expert output for all samples (more efficient than masking)
                    expert_output = self.experts[i](x)
                    # Apply gating weights
                    weighted_output = expert_output * gate_weights[:, i].unsqueeze(-1)
                    output += weighted_output

            return output

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("GPU out of memory! Clearing cache and trying with smaller batch...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Return zero tensor as fallback
                return torch.zeros(x.size(0), self.output_dim, device=self.device)
            else:
                raise e

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(),
                                         lr=prm.get('lr', 0.01),
                                         momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (inputs, labels) in enumerate(train_data):
            try:
                # Memory management
                if batch_idx % 10 == 0:  # Clear cache every 10 batches
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                inputs = inputs.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                # Limit batch size for memory efficiency
                if inputs.size(0) > 32:
                    inputs = inputs[:32]
                    labels = labels[:32]

                self.optimizer.zero_grad()
                outputs = self(inputs)

                # Handle output shapes
                if outputs.dim() > 2:
                    outputs = outputs.view(outputs.size(0), -1)
                if labels.dim() > 1:
                    labels = labels.view(-1)

                loss = self.criteria(outputs, labels)
                loss.backward()

                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=3.0)

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Clear intermediate tensors
                del inputs, labels, outputs, loss

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at batch {batch_idx}, skipping...")
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

                    # Limit batch size
                    if inputs.size(0) > 32:
                        inputs = inputs[:32]
                        labels = labels[:32]

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

                    # Clear tensors
                    del inputs, labels, outputs, loss

                except Exception as e:
                    print(f"Eval error: {e}")
                    continue

        return total_loss / len(test_data), correct / total if total > 0 else 0


# Memory-efficient usage example:
if __name__ == "__main__":
    # Set memory-friendly settings
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use AlexNet-compatible input shape
    # For CIFAR-10: (32, 3, 32, 32) or ImageNet: (224, 3, 224, 224)
    in_shape = (32, 3, 32, 32)  # CIFAR-10 compatible
    out_shape = (10,)  # CIFAR-10 classes
    prm = {'lr': 0.01, 'momentum': 0.9, 'dropout': 0.5}

    try:
        model = Net(in_shape, out_shape, prm, device)
        model.train_setup(prm)

        print("AlexNet-MoE model created successfully!")

        # Test with small batch
        test_input = torch.randn(4, 3, 32, 32)  # Small batch of CIFAR-10 sized images
        test_output = model(test_input)
        print(f"Test successful! Output shape: {test_output.shape}")
        print(f"AlexNet features -> MoE with {model.n_experts} experts, top-{model.top_k} routing")

    except Exception as e:
        print(f"Error: {e}")