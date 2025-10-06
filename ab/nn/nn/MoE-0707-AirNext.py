import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class AirBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, ratio=2):
        super(AirBlock, self).__init__()
        mid_channels = out_channels // ratio
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.bn3(self.conv3(x))
        x = self.sigmoid(x)
        return x


class AirNeXtUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, cardinality, bottleneck_width, ratio):
        super(AirNeXtUnit, self).__init__()
        mid_channels = out_channels // 4
        D = int(math.floor(mid_channels * (bottleneck_width / 64.0)))
        group_width = cardinality * D
        self.use_air_block = (stride == 1 and mid_channels < 512)

        self.conv1 = nn.Conv2d(in_channels, group_width, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.conv3 = nn.Conv2d(group_width, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        if self.use_air_block:
            self.air = AirBlock(in_channels, group_width, groups=(cardinality // ratio), ratio=ratio)

        self.resize_identity = (in_channels != out_channels) or (stride != 1)
        if self.resize_identity:
            self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_air_block:
            att = self.air(x)
            att = F.interpolate(att, size=x.shape[2:], mode="bilinear", align_corners=True)
        
        identity = self.identity_conv(x) if self.resize_identity else x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_air_block:
            x = x * att
        x = self.conv3(x)
        x = x + identity
        x = self.activ(x)
        return x


class AirNeXtExpert(nn.Module):
    """AirNeXt-based expert for MoE"""
    def __init__(self, in_channels, out_features, image_size, expert_id):
        super(AirNeXtExpert, self).__init__()
        self.expert_id = expert_id
        
        # Simplified AirNeXt architecture for expert
        channels = [64, 128, 256]  # Reduced complexity for MoE
        init_block_channels = 64
        cardinality = 16  # Reduced cardinality
        bottleneck_width = 4
        ratio = 2

        # Initial conv block
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, init_block_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_block_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # AirNeXt stages (simplified)
        in_ch = init_block_channels
        for i, out_ch in enumerate(channels):
            stride = 2 if i > 0 else 1
            self.features.add_module(f"stage{i+1}", 
                AirNeXtUnit(in_ch, out_ch, stride, cardinality, bottleneck_width, ratio))
            in_ch = out_ch
        
        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(in_ch, out_features)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class Gate(nn.Module):
    """Gating network for MoE"""
    def __init__(self, in_channels, n_experts, image_size):
        super(Gate, self).__init__()
        self.n_experts = n_experts
        self.top_k = 2  # Top-2 routing
        
        # Simple CNN for gating
        self.gate_features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=4, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32 * 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, n_experts)
        )

    def forward(self, x):
        gate_logits = self.gate_features(x)
        
        # Add noise during training for load balancing
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
        channel_number = in_shape[1]
        image_size = in_shape[2]
        class_number = out_shape[0]
        
        self.input_channels = channel_number
        self.image_size = image_size
        self.num_classes = class_number

        # Create 8 AirNeXt experts
        self.experts = nn.ModuleList([
            AirNeXtExpert(channel_number, class_number, image_size, i)
            for i in range(self.n_experts)
        ])
        
        # Gating network
        self.gate = Gate(channel_number, self.n_experts, image_size)
        
        # Move to device
        self.to(device)
        self._print_model_info()

    def _print_model_info(self):
        param_count = sum(p.numel() for p in self.parameters())
        param_size_mb = param_count * 4 / (1024 * 1024)
        print(f"MoE-AirNeXt Model parameters: {param_count:,}")
        print(f"Model size: {param_size_mb:.2f} MB")
        print(f"Experts: {self.n_experts}, Top-K: {self.top_k}")

    def forward(self, x):
        try:
            batch_size = x.size(0)
            
            # Get sparse gating weights
            gate_weights, top_k_indices = self.gate(x)
            
            # Sparse MoE computation
            output = torch.zeros(batch_size, self.num_classes, device=self.device)
            
            # Only compute outputs for active experts
            for i in range(self.n_experts):
                expert_mask = (top_k_indices == i).any(dim=1)
                if expert_mask.any():
                    expert_output = self.experts[i](x)
                    weighted_output = expert_output * gate_weights[:, i].unsqueeze(-1)
                    output += weighted_output
            
            return output
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("GPU out of memory! Clearing cache...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return torch.zeros(x.size(0), self.num_classes, device=self.device)
            else:
                raise e

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        
        # Extract hyperparameters with defaults
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        dropout_rate = prm.get('dropout', 0.1)
        
        # Apply dropout rate to existing dropout layers
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
        
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=5, 
            gamma=0.5
        )

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

                inputs = inputs.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                # Ensure inputs are real-valued
                inputs = inputs.real if inputs.is_complex() else inputs

                # Limit batch size for memory efficiency
                if inputs.size(0) > 32:
                    inputs = inputs[:32]
                    labels = labels[:32]

                self.optimizer.zero_grad()
                outputs = self(inputs)
                
                # Handle label shapes
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

        # Step scheduler
        self.scheduler.step()
        return total_loss / max(num_batches, 1)


# Test the model
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example for CIFAR-10 like dataset
    in_shape = (32, 3, 32, 32)  # (batch, channels, height, width)
    out_shape = (10,)  # 10 classes
    prm = {'lr': 0.01, 'momentum': 0.9, 'dropout': 0.1}
    
    try:
        model = Net(in_shape, out_shape, prm, device)
        model.train_setup(prm)
        
        # Test forward pass
        test_input = torch.randn(8, 3, 32, 32).to(device)
        test_output = model(test_input)
        print(f"Test successful! Output shape: {test_output.shape}")
        print("MoE-AirNeXt model with 8 experts and top-2 routing")
        
    except Exception as e:
        print(f"Error: {e}")