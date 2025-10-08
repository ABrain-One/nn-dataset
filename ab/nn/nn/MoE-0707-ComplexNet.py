import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Module, Conv2d, Linear
from torch.nn.functional import relu, max_pool2d
import gc


def supported_hyperparameters():
    return {'lr', 'momentum'}


# Complex number operations
def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real) - fi(input.imag)).type(dtype) \
        + 1j * (fr(input.imag) + fi(input.real)).type(dtype)


def complex_relu(input):
    return relu(input.real).type(torch.complex64) + 1j * relu(input.imag).type(torch.complex64)


def complex_max_pool2d(input, kernel_size, stride=None, padding=0,
                       dilation=1, ceil_mode=False, return_indices=False):
    def _retrieve_elements_from_indices(tensor, indices):
        flattened_tensor = tensor.flatten(start_dim=-2)
        output = flattened_tensor.gather(dim=-1, index=indices.flatten(start_dim=-2)).view_as(indices)
        return output
    
    absolute_value, indices = max_pool2d(
        input.abs(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True
    )
    absolute_value = absolute_value.type(torch.complex64)
    angle = torch.atan2(input.imag, input.real)
    angle = _retrieve_elements_from_indices(angle, indices)
    return absolute_value \
        * (torch.cos(angle).type(torch.complex64) + 1j * torch.sin(angle).type(torch.complex64))


# Complex layers
class ComplexConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return apply_complex(self.conv_r, self.conv_i, input)


class ComplexLinear(Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)


class ComplexBatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ComplexBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 3))
            self.bias = nn.Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype=torch.complex64))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            mean_r = input.real.mean([0, 2, 3]).type(torch.complex64)
            mean_i = input.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, :, None, None]

        if self.training or (not self.training and not self.track_running_stats):
            n = input.numel() / input.size(1)
            Crr = 1. / n * input.real.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1. / n * input.imag.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 0]
                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 1]
                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 2]

        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None, :, None, None] * input.real + Rri[None, :, None, None] * input.imag).type(torch.complex64) \
                + 1j * (Rii[None, :, None, None] * input.imag + Rri[None, :, None, None] * input.real).type(torch.complex64)

        if self.affine:
            input = (self.weight[None, :, 0, None, None] * input.real + self.weight[None, :, 2, None, None] * input.imag + \
                     self.bias[None, :, 0, None, None]).type(torch.complex64) \
                    + 1j * (self.weight[None, :, 2, None, None] * input.real + self.weight[None, :, 1, None, None] * input.imag + \
                            self.bias[None, :, 1, None, None]).type(torch.complex64)

        return input


# Complex Expert for MoE
class ComplexExpert(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim):
        super(ComplexExpert, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Complex linear layers
        self.fc1 = ComplexLinear(feature_dim, hidden_dim)
        self.fc2 = ComplexLinear(hidden_dim, hidden_dim // 2)
        self.fc3 = ComplexLinear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Ensure input is complex
        if not torch.is_complex(x):
            x = x.type(torch.complex64)
        
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Adjust input dimension if needed
        if x.size(-1) != self.feature_dim:
            if x.size(-1) > self.feature_dim:
                x = x[:, :self.feature_dim]
            else:
                padding = self.feature_dim - x.size(-1)
                # Pad with zeros for complex numbers
                x = F.pad(x, (0, padding))

        x = complex_relu(self.fc1(x))
        # Apply dropout to both real and imaginary parts
        x_real = self.dropout(x.real)
        x_imag = self.dropout(x.imag)
        x = x_real.type(torch.complex64) + 1j * x_imag.type(torch.complex64)
        
        x = complex_relu(self.fc2(x))
        x_real = self.dropout(x.real)
        x_imag = self.dropout(x.imag)
        x = x_real.type(torch.complex64) + 1j * x_imag.type(torch.complex64)
        
        x = self.fc3(x)
        return x


# Gate network (remains real-valued for routing decisions)
class ComplexGate(nn.Module):
    def __init__(self, feature_dim, n_experts, hidden_dim=32):
        super(ComplexGate, self).__init__()
        self.feature_dim = feature_dim
        self.n_experts = n_experts
        self.top_k = 2  # Top-2 routing
        
        # Gate operates on magnitude of complex features
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_experts)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Convert complex input to magnitude for gating decision
        if torch.is_complex(x):
            x = x.abs()  # Use magnitude for routing
        
        x = x.float()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        if x.size(-1) != self.feature_dim:
            if x.size(-1) > self.feature_dim:
                x = x[:, :self.feature_dim]
            else:
                padding = self.feature_dim - x.size(-1)
                x = F.pad(x, (0, padding))

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        gate_logits = self.fc2(x)
        
        # Add noise for load balancing during training
        if self.training:
            noise = torch.randn_like(gate_logits) * 0.1
            gate_logits = gate_logits + noise
        
        # Top-k gating
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
        self.n_experts = 8
        self.top_k = 2
        
        # Extract shape information
        self.in_channels = in_shape[1] if len(in_shape) > 1 else 1
        self.in_height = in_shape[2] if len(in_shape) > 2 else 28
        self.in_width = in_shape[3] if len(in_shape) > 3 else 28
        self.output_dim = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape
        
        # Complex convolutional layers
        self.conv1 = ComplexConv2d(self.in_channels, 10, 5, 1)
        self.bn = ComplexBatchNorm2d(10)
        self.conv2 = ComplexConv2d(10, 20, 5, 1)
        
        # Calculate feature dimension after conv layers
        with torch.no_grad():
            # Create dummy complex input
            tmp_input = torch.zeros(1, self.in_channels, self.in_height, self.in_width).type(torch.complex64)
            tmp_features = self._extract_features(tmp_input)
            self.feature_dim = tmp_features.view(-1).size(0)
        
        # Memory-efficient hidden dimension
        self.hidden_dim = min(128, max(32, self.feature_dim // 16))
        
        # Create complex experts
        self.experts = nn.ModuleList([
            ComplexExpert(self.feature_dim, self.hidden_dim, self.output_dim)
            for _ in range(self.n_experts)
        ])
        
        # Gate network
        self.gate = ComplexGate(self.feature_dim, self.n_experts, self.hidden_dim // 2)
        
        # Move to device
        self.to(device)
        
        # Print model info
        self._print_memory_info()

    def _extract_features(self, x):
        """Extract features using complex convolutions"""
        x = x.view(-1, self.in_channels, self.in_height, self.in_width)
        x = self.conv1(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = self.bn(x)
        x = complex_relu(self.conv2(x))
        x = complex_max_pool2d(x, 2, 2)
        return x

    def _print_memory_info(self):
        param_count = sum(p.numel() for p in self.parameters())
        param_size_mb = param_count * 8 / (1024 * 1024)  # 8 bytes per complex64
        print(f"ComplexMoE-{self.n_experts} Model parameters: {param_count:,}")
        print(f"Model size: {param_size_mb:.2f} MB")
        print(f"Experts: {self.n_experts}, Top-K: {self.top_k}")
        print(f"Feature dim: {self.feature_dim}, Hidden dim: {self.hidden_dim}")

    def forward(self, x):
        try:
            # Convert to complex if needed
            if not torch.is_complex(x):
                x = x.type(torch.complex64)
            
            batch_size = x.size(0)
            
            # Extract complex features
            features = self._extract_features(x)
            features_flat = features.view(batch_size, -1)
            
            # Get gating decisions
            gate_weights, top_k_indices = self.gate(features_flat)
            
            # Sparse MoE computation with complex experts
            output = torch.zeros(batch_size, self.output_dim, device=self.device, dtype=torch.complex64)
            
            # Only compute outputs for active experts
            for i in range(self.n_experts):
                expert_mask = (top_k_indices == i).any(dim=1)
                if expert_mask.any():
                    expert_output = self.experts[i](features_flat)
                    # Apply gating weights (real weights on complex outputs)
                    weighted_output = expert_output * gate_weights[:, i].unsqueeze(-1).type(torch.complex64)
                    output += weighted_output
            
            # Convert to real for final classification
            output = output.abs()  # Use magnitude for classification
            output = F.log_softmax(output, dim=1)
            
            return output

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("GPU out of memory! Clearing cache...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
                if batch_idx % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                inputs = inputs.to(self.device)
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
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

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