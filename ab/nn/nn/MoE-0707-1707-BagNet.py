import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SimpleExpert(nn.Module):
    """Simple but effective expert for CIFAR-10"""
    def __init__(self, in_channels, num_classes, expert_id=0):
        super().__init__()
        self.expert_id = expert_id
        
        # Different channel configurations for diversity
        base_channels = [32, 48, 64, 48, 32, 56, 40, 44][expert_id]
        
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # Use residual blocks for better gradient flow
        self.layer1 = self._make_layer(base_channels, base_channels, 2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SimpleGate(nn.Module):
    def __init__(self, in_channels, n_experts):
        super().__init__()
        self.n_experts = n_experts
        
        # Very simple but effective gating
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(in_channels * 16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_experts)
        )
        
        self.top_k = min(2, n_experts)  # Ensure top_k doesn't exceed n_experts
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        gate_logits = self.gate(x)
        
        # Simple softmax gating (more stable than top-k for small experts)
        if self.n_experts <= 4:
            # Use all experts for small number
            gates = F.softmax(gate_logits, dim=-1)
            top_k_indices = torch.arange(self.n_experts, device=x.device).expand(x.size(0), -1)
        else:
            # Top-k for larger number of experts
            top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
            top_k_gates = F.softmax(top_k_logits, dim=-1)
            
            gates = torch.zeros_like(gate_logits)
            gates.scatter_(1, top_k_indices, top_k_gates)
        
        return gates, top_k_indices


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        
        # Extract parameters
        self.in_channels = in_shape[1]
        self.img_size = in_shape[2]
        self.num_classes = out_shape[0]
        
        # Reduce number of experts for better training with limited epochs
        self.n_experts = 4  # Reduced from 8
        
        # Store hyperparameters with scaling
        self.learning_rate = max(prm['lr'] * 100, 0.001)  # Scale up learning rate
        self.momentum = prm['momentum']
        self.dropout = prm.get('dropout', 0.1)
        
        print(f"Scaled learning rate: {self.learning_rate}")
        
        # Create experts
        self.experts = nn.ModuleList([
            SimpleExpert(self.in_channels, self.num_classes, expert_id=i)
            for i in range(self.n_experts)
        ])
        
        # Simple gating
        self.gate = SimpleGate(self.in_channels, self.n_experts)
        
        # Add a simple baseline classifier for better initialization
        self.baseline = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, self.num_classes)
        )
        
        # Move to device
        self.to(device)
        
        # Print model info
        self._print_model_info()

    def _print_model_info(self):
        param_count = sum(p.numel() for p in self.parameters())
        param_size_mb = param_count * 4 / (1024 * 1024)
        print(f"MoE Model parameters: {param_count:,}")
        print(f"Model size: {param_size_mb:.2f} MB")
        print(f"Experts: {self.n_experts}")
        print(f"Learning rate: {self.learning_rate}")

    def forward(self, x):
        # Get gating weights
        gate_weights, _ = self.gate(x)
        
        # Get baseline output for stability
        baseline_output = self.baseline(x)
        
        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        # Combine expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)
        gate_weights_expanded = gate_weights.unsqueeze(-1)
        moe_output = torch.sum(expert_outputs * gate_weights_expanded, dim=1)
        
        # Combine with baseline for better stability (ensemble approach)
        output = 0.7 * moe_output + 0.3 * baseline_output
        
        return output

    def train_setup(self, prm):
        """Required by LEMUR framework"""
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss()
        
        # Use SGD with scaled learning rate and momentum from framework
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=prm['momentum'],
            weight_decay=5e-4,
            nesterov=True
        )

    def learn(self, train_data):
        """Required by LEMUR framework"""
        self.train()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0

        for batch_idx, (inputs, labels) in enumerate(train_data):
            try:
                inputs = inputs.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                self.optimizer.zero_grad()
                outputs = self(inputs)
                
                if labels.dim() > 1:
                    labels = labels.view(-1)

                loss = self.criteria(outputs, labels)
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Track metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                num_batches += 1

                # Progress tracking
                if batch_idx % 200 == 0 and batch_idx > 0:
                    acc = 100. * correct / total
                    print(f'Batch {batch_idx}/{len(train_data)}, Loss: {loss.item():.4f}, Acc: {acc:.2f}%')

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

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct / total if total > 0 else 0
        print(f'Final Training - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
        
        return avg_loss

    def evaluate(self, test_data):
        """Evaluation method"""
        self.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_data:
                try:
                    inputs = inputs.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.long)

                    outputs = self(inputs)

                    if labels.dim() > 1:
                        labels = labels.view(-1)

                    loss = self.criteria(outputs, labels)
                    total_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    del inputs, labels, outputs, loss

                except Exception as e:
                    continue

        avg_loss = total_loss / len(test_data) if len(test_data) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        print(f'Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
        
        return avg_loss, accuracy


# Example usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    in_shape = (32, 3, 32, 32)
    out_shape = (10,)
    prm = {'lr': 0.01, 'momentum': 0.9, 'dropout': 0.1}
    
    try:
        model = Net(in_shape, out_shape, prm, device)
        model.train_setup(prm)
        
        print("Model created successfully!")
        
        # Test forward pass
        test_input = torch.randn(8, 3, 32, 32).to(device)
        test_output = model(test_input)
        print(f"Output shape: {test_output.shape}")
        
    except Exception as e:
        print(f"Error: {e}")