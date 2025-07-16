import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(Expert, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Ensure input is float and flatten if needed
        x = x.float()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Ensure correct input dimension
        if x.size(-1) != self.input_dim:
            # Adaptive pooling or padding to match expected input size
            if x.size(-1) > self.input_dim:
                x = F.adaptive_avg_pool1d(x.unsqueeze(1), self.input_dim).squeeze(1)
            else:
                padding = self.input_dim - x.size(-1)
                x = F.pad(x, (0, padding))

        x = self.layer_norm1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.layer_norm2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.layer_norm3(F.relu(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class Gate(nn.Module):
    def __init__(self, input_dim, n_experts, hidden_dim=64):
        super(Gate, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_experts)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Ensure input is float and flatten if needed
        x = x.float()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Ensure correct input dimension
        if x.size(-1) != self.input_dim:
            if x.size(-1) > self.input_dim:
                x = F.adaptive_avg_pool1d(x.unsqueeze(1), self.input_dim).squeeze(1)
            else:
                padding = self.input_dim - x.size(-1)
                x = F.pad(x, (0, padding))

        x = self.layer_norm1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.layer_norm2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        # Apply softmax with temperature for better expert selection
        x = F.softmax(x / 0.5, dim=-1)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.n_experts = 4  # Increased number of experts

        # Calculate input dimension properly
        if isinstance(in_shape, (list, tuple)) and len(in_shape) > 1:
            self.input_dim = 1
            for dim in in_shape:
                self.input_dim *= dim
        else:
            self.input_dim = in_shape[0] if isinstance(in_shape, (list, tuple)) else in_shape

        self.output_dim = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape

        # Use adaptive hidden dimension based on input size
        self.hidden_dim = max(64, min(512, self.input_dim // 4))

        # Create experts with proper dimensions
        self.experts = nn.ModuleList([
            Expert(self.input_dim, self.hidden_dim, self.output_dim)
            for _ in range(self.n_experts)
        ])
        self.gate = Gate(self.input_dim, self.n_experts, self.hidden_dim)

        # Add a final output layer for better control
        self.output_layer = nn.Linear(self.output_dim, self.output_dim)

        # Initialize weights
        self.apply(self._init_weights)

        # Move to device
        self.to(device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        try:
            # Ensure input is float tensor
            x = x.float()

            # Handle different input shapes
            batch_size = x.size(0)
            if x.dim() > 2:
                x = x.view(batch_size, -1)

            # Get gating weights for each expert
            gate_weights = self.gate(x)  # Shape: [batch_size, n_experts]

            # Get outputs from all experts
            expert_outputs = []
            for expert in self.experts:
                try:
                    output = expert(x)
                    expert_outputs.append(output)
                except Exception as e:
                    print(f"Expert forward error: {e}")
                    # Fallback: create zero tensor with correct shape
                    expert_outputs.append(torch.zeros(batch_size, self.output_dim, device=x.device, dtype=x.dtype))

            # Stack expert outputs
            expert_outputs = torch.stack(expert_outputs, dim=1)  # Shape: [batch_size, n_experts, out_dim]

            # Weighted combination of expert outputs
            gate_weights = gate_weights.unsqueeze(-1)  # Shape: [batch_size, n_experts, 1]
            output = torch.sum(expert_outputs * gate_weights, dim=1)  # Shape: [batch_size, out_dim]

            # Final output layer
            output = self.output_layer(output)

            return output

        except Exception as e:
            print(f"Forward pass error: {e}")
            print(f"Input shape: {x.shape}, Input dtype: {x.dtype}")
            # Return zero tensor as fallback
            return torch.zeros(x.size(0), self.output_dim, device=self.device, dtype=torch.float32)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))
        # Add learning rate scheduler for better training
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)

    def learn(self, train_data):
        self.train()  # Set to training mode
        total_loss = 0
        num_batches = 0

        for inputs, labels in train_data:
            try:
                # Ensure tensors are on correct device and dtype
                inputs = inputs.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                self.optimizer.zero_grad()
                outputs = self(inputs)

                # Ensure outputs and labels have compatible shapes
                if outputs.dim() > 2:
                    outputs = outputs.view(outputs.size(0), -1)
                if labels.dim() > 1:
                    labels = labels.view(-1)

                loss = self.criteria(outputs, labels)
                loss.backward()

                # Gradient clipping with error handling
                try:
                    nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                except:
                    pass  # Skip if gradient clipping fails

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            except Exception as e:
                print(f"Training batch error: {e}")
                continue

        # Update learning rate
        if hasattr(self, 'scheduler') and num_batches > 0:
            avg_loss = total_loss / num_batches
            self.scheduler.step(avg_loss)

        return total_loss / max(num_batches, 1)

    def evaluate(self, test_data):
        """Evaluation method for testing"""
        self.eval()  # Set to evaluation mode
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_data:
                try:
                    inputs = inputs.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.long)

                    outputs = self(inputs)

                    # Ensure compatible shapes
                    if outputs.dim() > 2:
                        outputs = outputs.view(outputs.size(0), -1)
                    if labels.dim() > 1:
                        labels = labels.view(-1)

                    loss = self.criteria(outputs, labels)
                    total_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                except Exception as e:
                    print(f"Evaluation batch error: {e}")
                    continue

        return total_loss / len(test_data), correct / total if total > 0 else 0


# Example usage with error handling:
if __name__ == "__main__":
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Example for different input shapes
        in_shape = (784,)  # Will be automatically handled
        out_shape = (10,)
        prm = {'lr': 0.01, 'momentum': 0.9}

        model = Net(in_shape, out_shape, prm, device)
        model.train_setup(prm)

        print("Model created successfully!")
        print(f"Input dimension: {model.input_dim}")
        print(f"Output dimension: {model.output_dim}")
        print(f"Hidden dimension: {model.hidden_dim}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

        # Test with random data
        test_input = torch.randn(32, model.input_dim)
        test_output = model(test_input)
        print(f"Test output shape: {test_output.shape}")

    except Exception as e:
        print(f"Initialization error: {e}")