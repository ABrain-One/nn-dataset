import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class DarkNetUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pointwise: bool, alpha: float):
        super(DarkNetUnit, self).__init__()
        self.activation = nn.LeakyReLU(negative_slope=alpha, inplace=True)
        if pointwise:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DenseNetExpert(nn.Module):
    """Expert network based on DenseNet architecture"""
    def __init__(self, in_channels, num_classes, channels_config, alpha=0.1):
        super(DenseNetExpert, self).__init__()
        self.features = nn.Sequential()
        
        for i, channels_per_stage in enumerate(channels_config):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                pointwise = (len(channels_per_stage) > 1) and not (((j + 1) % 2 == 1) ^ True)
                stage.add_module(f"unit{j + 1}", DarkNetUnit(in_channels, out_channels, pointwise, alpha))
                in_channels = out_channels
            if i != len(channels_config) - 1:
                stage.add_module(f"pool{i + 1}", nn.MaxPool2d(kernel_size=2, stride=2))
            self.features.add_module(f"stage{i + 1}", stage)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1),
            nn.LeakyReLU(negative_slope=alpha, inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x


class Gate(nn.Module):
    def __init__(self, in_channels, image_size, n_experts, hidden_dim=32):
        super(Gate, self).__init__()
        self.n_experts = n_experts
        self.top_k = 2  # Top-2 routing
        
        # Adaptive pooling to fixed size for gating
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        gate_input_dim = in_channels * 16  # 4x4 feature map
        
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_experts)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Pool and flatten for gating decision
        pooled = self.adaptive_pool(x)
        gate_input = pooled.view(batch_size, -1)
        
        gate_logits = self.gate_net(gate_input)
        
        # Add noise for load balancing during training
        if self.training:
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

        # Extract input parameters
        in_channels = in_shape[1]
        image_size = in_shape[2]
        num_classes = out_shape[0]
        alpha = 0.1

        # Different channel configurations for expert diversity
        expert_configs = [
            [[32, 32], [64, 64], [128, 128]],      # Lightweight expert 1
            [[64, 64], [128, 128], [256, 256]],    # Medium expert 1
            [[48, 48], [96, 96], [192, 192]],      # Lightweight expert 2
            [[64, 64, 64], [128, 128], [256]],     # Deep-narrow expert
            [[96, 96], [192, 192], [384, 384]],    # Medium expert 2
            [[64], [128, 128, 128], [256, 256]],   # Wide expert
            [[80, 80], [160, 160], [320, 320]],    # Medium expert 3
            [[64, 64], [128, 128, 128], [256]]     # Mixed expert
        ]

        # Create 8 DenseNet-based experts with different architectures
        self.experts = nn.ModuleList([
            DenseNetExpert(in_channels, num_classes, config, alpha)
            for config in expert_configs
        ])
        
        # Gate for top-2 routing
        self.gate = Gate(in_channels, image_size, self.n_experts, hidden_dim=64)

        # Move to device
        self.to(device)
        self._print_memory_info()

    def _print_memory_info(self):
        param_count = sum(p.numel() for p in self.parameters())
        param_size_mb = param_count * 4 / (1024 * 1024)
        print(f"MoE-DenseNet Model parameters: {param_count:,}")
        print(f"Model size: {param_size_mb:.2f} MB")
        print(f"Experts: {self.n_experts}, Top-K: {self.top_k}")

        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

    def forward(self, x):
        try:
            batch_size = x.size(0)
            
            # Limit batch size if too large
            if batch_size > 64:
                print(f"Warning: Large batch size {batch_size}, consider reducing")

            # Get sparse gating weights (top-2)
            gate_weights, top_k_indices = self.gate(x)

            # Sparse MoE computation - only compute active experts
            output = None
            
            for i in range(self.n_experts):
                # Check if this expert is used by any sample
                expert_mask = (top_k_indices == i).any(dim=1)
                if expert_mask.any():
                    # Get expert output
                    expert_output = self.experts[i](x)
                    
                    # Apply gating weights
                    weighted_output = expert_output * gate_weights[:, i].unsqueeze(-1)
                    
                    if output is None:
                        output = weighted_output
                    else:
                        output += weighted_output

            return output if output is not None else torch.zeros(batch_size, self.experts[0].output[0].out_channels, device=self.device)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("GPU out of memory! Clearing cache...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return torch.zeros(x.size(0), self.num_classes, device=self.device)
            else:
                raise e

    def train_setup(self, prm: dict):
        self.to(self.device)
        learning_rate = float(prm.get("lr", 0.01))
        momentum = float(prm.get("momentum", 0.9))
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

    def learn(self, train_data):
        self.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_data):
            try:
                # Memory management
                if batch_idx % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Limit batch size for memory efficiency
                if inputs.size(0) > 32:
                    inputs = inputs[:32]
                    targets = targets[:32]

                self.optimizer.zero_grad()
                outputs = self(inputs)
                
                loss = self.criteria(outputs, targets)
                loss.backward()

                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Clear intermediate tensors
                del inputs, targets, outputs, loss

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
            for inputs, targets in test_data:
                try:
                    # Ensure inputs are real-valued float tensors
                    if inputs.is_complex():
                        inputs = inputs.real
                    inputs = inputs.to(self.device, dtype=torch.float32)
                    targets = targets.to(self.device)

                    if inputs.size(0) > 32:
                        inputs = inputs[:32]
                        targets = targets[:32]

                    outputs = self(inputs)
                    loss = self.criteria(outputs, targets)
                    total_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                    del inputs, targets, outputs, loss

                except Exception as e:
                    print(f"Eval error: {e}")
                    continue

        return total_loss / len(test_data), correct / total if total > 0 else 0


if __name__ == "__main__":
    # Memory-friendly settings
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Example for CIFAR-10 like input
    in_shape = (32, 3, 32, 32)  # (batch, channels, height, width)
    out_shape = (10,)           # 10 classes
    prm = {'lr': 0.01, 'momentum': 0.9}

    try:
        model = Net(in_shape, out_shape, prm, device)
        model.train_setup(prm)

        print("MoE-DenseNet model created successfully!")

        # Test with small batch
        test_input = torch.randn(8, 3, 32, 32).to(device)
        test_output = model(test_input)
        print(f"Test successful! Output shape: {test_output.shape}")
        print(f"Sparse MoE with {model.n_experts} DenseNet experts, top-{model.top_k} routing")

    except Exception as e:
        print(f"Error: {e}")