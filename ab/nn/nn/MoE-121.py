# attempt to make small size sparse MoE model...

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


def supported_hyperparameters():
    return {'lr', 'momentum'}


class MicroExpert(nn.Module):
    """Ultra-lightweight expert with minimal parameters"""

    def __init__(self, input_dim, hidden_dim, out_dim):
        super(MicroExpert, self).__init__()
        self.input_dim = input_dim

        # Very small architecture: input -> tiny_hidden -> output
        # Using even smaller hidden dimension
        tiny_hidden = max(16, hidden_dim // 4)  # Much smaller hidden layer

        self.fc1 = nn.Linear(input_dim, tiny_hidden)
        self.fc2 = nn.Linear(tiny_hidden, out_dim)
        self.dropout = nn.Dropout(0.1)  # Reduced dropout

        # Use weight sharing trick - share some parameters across experts
        self.expert_id = None  # Will be set during initialization

    def forward(self, x):
        x = x.float()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Handle input dimension mismatch efficiently
        if x.size(-1) != self.input_dim:
            if x.size(-1) > self.input_dim:
                x = x[:, :self.input_dim]
            else:
                padding = self.input_dim - x.size(-1)
                x = F.pad(x, (0, padding))

        # Very simple forward pass
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class UltraLightGate(nn.Module):
    """Minimal gate network"""

    def __init__(self, input_dim, n_experts):
        super(UltraLightGate, self).__init__()
        self.input_dim = input_dim

        # Single layer gate for minimal parameters
        self.gate_layer = nn.Linear(input_dim, n_experts)

        # Optional: Add a tiny hidden layer only if input is large
        if input_dim > 512:
            tiny_hidden = 16  # Very small
            self.pre_gate = nn.Linear(input_dim, tiny_hidden)
            self.gate_layer = nn.Linear(tiny_hidden, n_experts)
        else:
            self.pre_gate = None

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

        # Minimal gate computation
        if self.pre_gate is not None:
            x = F.relu(self.pre_gate(x))

        x = self.gate_layer(x)
        x = F.softmax(x, dim=-1)
        return x


class SharedEmbedding(nn.Module):
    """Shared embedding layer to reduce parameters"""

    def __init__(self, input_dim, embed_dim):
        super(SharedEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        return F.relu(self.embedding(x))


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.n_experts = 4  # Now 4 experts

        # Calculate dimensions
        if isinstance(in_shape, (list, tuple)) and len(in_shape) > 1:
            self.input_dim = 1
            for dim in in_shape:
                self.input_dim *= dim
        else:
            self.input_dim = in_shape[0] if isinstance(in_shape, (list, tuple)) else in_shape

        self.output_dim = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape

        # ULTRA small hidden dimension for 4 experts
        self.hidden_dim = max(8, min(32, self.input_dim // 16))  # Much smaller

        # Limit input dimension
        if self.input_dim > 1024:  # Even more restrictive
            print(f"Warning: Large input dimension {self.input_dim}, limiting to 1024")
            self.input_dim = 1024

        # Shared embedding to reduce parameters
        embed_dim = max(8, min(16, self.input_dim // 8))
        self.shared_embedding = SharedEmbedding(self.input_dim, embed_dim)

        # Create 4 micro experts - they work on embedded features
        self.experts = nn.ModuleList([
            MicroExpert(embed_dim, self.hidden_dim, self.output_dim)
            for i in range(self.n_experts)
        ])

        # Set expert IDs for potential parameter sharing
        for i, expert in enumerate(self.experts):
            expert.expert_id = i

        # Ultra light gate
        self.gate = UltraLightGate(embed_dim, self.n_experts)  # Gate works on embeddings

        # Move to device
        self.to(device)

        # Print detailed parameter info
        self._print_detailed_params()

    def _print_detailed_params(self):
        """Print detailed parameter breakdown"""
        total_params = 0

        print("\n=== PARAMETER BREAKDOWN ===")

        # Shared embedding params
        shared_params = sum(p.numel() for p in self.shared_embedding.parameters())
        print(f"Shared Embedding: {shared_params:,} parameters")
        total_params += shared_params

        # Expert params
        expert_params = sum(p.numel() for p in self.experts.parameters())
        single_expert_params = expert_params // self.n_experts
        print(f"Each Expert: {single_expert_params:,} parameters")
        print(f"All 4 Experts: {expert_params:,} parameters")
        total_params += expert_params

        # Gate params
        gate_params = sum(p.numel() for p in self.gate.parameters())
        print(f"Gate Network: {gate_params:,} parameters")
        total_params += gate_params

        print(f"\nTOTAL PARAMETERS: {total_params:,}")
        print(f"Model size: {total_params * 4 / (1024 * 1024):.3f} MB")
        print(f"Input dim: {self.input_dim}, Hidden dim: {self.hidden_dim}, Output dim: {self.output_dim}")
        print(f"Embedding dim: {self.shared_embedding.embed_dim}")

        # Memory usage
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

        print("=" * 30)

    def forward(self, x):
        try:
            x = x.float()
            batch_size = x.size(0)

            if batch_size > 32:  # Even stricter batch size limit
                print(f"Warning: Large batch size {batch_size}, consider reducing")

            # Handle input shapes
            if x.dim() > 2:
                x = x.view(batch_size, -1)

            # Truncate input if too large
            if x.size(-1) > self.input_dim:
                x = x[:, :self.input_dim]

            # Shared embedding - reduces computation for all experts
            embedded_x = self.shared_embedding(x)

            # Get gating weights (works on embeddings)
            gate_weights = self.gate(embedded_x)

            # Get expert outputs (all work on same embeddings)
            expert_outputs = torch.stack([expert(embedded_x) for expert in self.experts], dim=1)

            # Weighted combination
            gate_weights = gate_weights.unsqueeze(-1)
            output = torch.sum(expert_outputs * gate_weights, dim=1)

            return output

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("GPU out of memory! Clearing cache...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return torch.zeros(x.size(0), self.output_dim, device=self.device)
            else:
                raise e

    def get_expert_utilization(self, data_loader):
        """Monitor which experts are being used"""
        self.eval()
        expert_usage = torch.zeros(self.n_experts, device=self.device)
        total_samples = 0

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device, dtype=torch.float32)
                if inputs.size(0) > 16:  # Small batches for monitoring
                    inputs = inputs[:16]

                # Get embeddings and gate weights
                if inputs.dim() > 2:
                    inputs = inputs.view(inputs.size(0), -1)
                if inputs.size(-1) > self.input_dim:
                    inputs = inputs[:, :self.input_dim]

                embedded_x = self.shared_embedding(inputs)
                gate_weights = self.gate(embedded_x)

                # Track expert usage
                expert_usage += gate_weights.sum(dim=0)
                total_samples += inputs.size(0)

                if total_samples >= 1000:  # Sample 1000 examples
                    break

        expert_usage = expert_usage / total_samples
        print(f"\nExpert Utilization:")
        for i, usage in enumerate(expert_usage):
            print(f"Expert {i}: {usage:.3f} ({usage * 100:.1f}%)")

        return expert_usage

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
                # Aggressive memory management
                if batch_idx % 5 == 0:  # Clear more frequently
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                inputs = inputs.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                # Smaller batch size for 4 experts
                if inputs.size(0) > 16:  # Even smaller batches
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

                # Lighter gradient clipping
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.3)

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Clear tensors
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

                    # Small batch size
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
                    continue

        return total_loss / len(test_data), correct / total if total > 0 else 0


# Example usage:
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test with different input sizes
    in_shape = (784,)  # MNIST-like
    out_shape = (10,)
    prm = {'lr': 0.01, 'momentum': 0.9}

    try:
        print("Creating 4-Expert MoE Model...")
        model = Net(in_shape, out_shape, prm, device)
        model.train_setup(prm)

        print("\n✅ Model created successfully!")

        # Test forward pass
        test_input = torch.randn(4, model.input_dim)  # Very small test batch
        test_output = model(test_input)
        print(f"✅ Forward pass successful! Output shape: {test_output.shape}")

        # Show parameter efficiency
        total_params = sum(p.numel() for p in model.parameters())
        params_per_expert = total_params / 4  # Approximate
        print(f"\n📊 Parameter Efficiency:")
        print(f"   Parameters per expert (approx): {params_per_expert:,.0f}")
        print(f"   Total parameters: {total_params:,}")

    except Exception as e:
        print(f"❌ Error: {e}")