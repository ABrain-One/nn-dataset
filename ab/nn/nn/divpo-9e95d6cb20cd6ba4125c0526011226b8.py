import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.module_1 = Block(32, 32)
        self.module_2 = Block(32, 32)
        self.module_3 = Block(32, 64)
        self.module_4 = Block(64, 64)
        self.module_5 = Block(64, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, x):
        # Optimized forward pass for minimum resource usage
        x = self.stem(x)
        x = self.module_1(x)
        x = self.module_2(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.module_3(x)
        x = self.module_4(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.module_5(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0001)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# Specify your target number of parameters here (e.g. 150000 <= 213201 <= 500000)
PARAMETER_TARGET = 213201

# Derive the architecture based on the target parameter count and under the total parameter limit
under_limit = PARAMETER_TARGET <= 500000
ratio_under = PARAMETER_TARGET / 213201

# Estimate the influence of the first blocks on the total, round down to an integer
first_block_multiplier = int(2.768 * ratio_under ** 1.257) if under_limit else 17

# Estimate the final layer's input features by fitting the formula
first_block_out_features = [   48,   48,   48,   48,   96,   96,  128,  128,  160,  160,  192,  192,  224,  224,  256]
self_estimated_out_features = first_block_out_features[first_block_multiplier - 1]

# Calculate the remaining parameter count to fit under the limit
PARAMETER_LIMIT = first_block_multiplier * 213201

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return self.layer(x)

    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}'

    def get_out_channels(self):
        return self.out_channels

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())

    def estimate_out_features(self):
        return self_estimated_out_features

    def check_budget(self, remaining_params: int):
        return remaining_params >= self.get_total_params()

    @staticmethod
    def compute_out_features(in_features, block):
        # Estimate the output features of a block (without running it)
        dummy = Block(in_features, in_features)
        dummy_out_features = dummy.estimate_out_features()
        return dummy_out_features

    @staticmethod
    def grow_to_capacity(in_features, block, remaining_params: int, prune_id=-1, fixed_out=None):
        # Attempt to grow the block to fit within the remaining parameters
        out_features = Block.compute_out_features(in_features, block)
        while out_features > in_features and block.check_budget(remaining_params):
            if fixed_out is not None:
                out_features = fixed_out
            elif prune_id == 0:
                block.layer[-1] = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, bias=False)
            elif prune_id == 1:
                block.layer = nn.Sequential(
                    block.layer[0],
                    nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, bias=False)
                )
            else:
                block.layer[-prune_id] = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, bias=False)
            out_features = block.estimate_out_features()
        return out_features

    @classmethod
    def from_channels(cls, in_channels, out_channels, prm, device, remaining_params: int, prune_id=-1, fixed_out=None):
        block = cls(in_channels, out_channels)
        out_features = Block.grow_to_capacity(in_channels, block, remaining_params, prune_id, fixed_out)
        block.in_channels = out_features
        block.out_channels = out_features
        return block, block.get_total_params()