import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class FractalDropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        if not self.training: 
            return torch.stack(inputs).mean(dim=0)
        n = len(inputs)
        mask = torch.bernoulli(torch.full((n,), 1 - self.drop_prob, device=inputs[0].device))
        if mask.sum() == 0: 
            mask[torch.randint(0, n, (1,)).item()] = 1.0
        active = [inp for inp, m in zip(inputs, mask) if m > 0]
        return torch.stack(active).mean(dim=0)

class FractalBlock(nn.Module):
    def __init__(self, n_columns: int, channels: int, dropout_prob: float):
        super().__init__()
        self.n_columns = int(n_columns)
        channels = int(channels)

        activation_layer = nn.SiLU(inplace=True)
        conv_layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )
        norm_layer = nn.BatchNorm2d(channels)

        self.conv = nn.Sequential(
            conv_layer,
            norm_layer,
            activation_layer
        )

        if self.n_columns > 1:
            self.left = FractalBlock(self.n_columns - 1, channels, dropout_prob)
            self.right_1 = FractalBlock(self.n_columns - 1, channels, dropout_prob)
            self.right_2 = FractalBlock(self.n_columns - 1, channels, dropout_prob)
            self.join = FractalDropPath(drop_prob=dropout_prob)

    def forward(self, x):
        if self.n_columns == 1: 
            return self.conv(x)
        out_left = self.left(x)
        out_right = self.right_2(self.right_1(x))
        return self.join([out_left, out_right])

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.prm = prm
        self.lr = prm['lr']
        self.batch_size = prm['batch']
        self.transform = prm['transform']
        
        in_channels = in_shape[1]
        out_classes = out_shape[0]
        
        # 1. Entry Block: 5x5 conv to capture wider context immediately (borrowed from Model 2's strong entry)
        # Channels: 96 (avoiding 64-128-256-512 progression)
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )
        
        # 2. Feature Extraction: Fractal Blocks (borrowed from Model 1)
        # We use a shallow but wide fractal structure.
        # Stage 1: 96 channels, 32x32 spatial
        self.block1 = FractalBlock(3, 96, 0.0)
        self.pool1 = nn.AvgPool2d(2)
        
        # Transition to 192 channels (192 is a "non-standard" width, helps with diversity)
        self.trans1 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 192 channels, 16x16 spatial
        self.block2 = FractalBlock(3, 192, 0.0)
        self.pool2 = nn.AvgPool2d(2)
        
        # Transition to 384 channels
        self.trans2 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 384 channels, 8x8 spatial
        self.block3 = FractalBlock(3, 384, 0.0)
        
        # 3. Classification Head
        # We use Global Average Pooling to reduce parameters and regularization needs.
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(384, out_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Apply horizontal flip if transform specifies it
        # This acts as an on-the-fly data augmentation during training
        if self.training and self.transform == 'norm_32_flip':
            # Randomly flip 50% of the batch
            flip_mask = torch.rand(x.size(0), device=x.device) > 0.5
            if flip_mask.any():
                x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
        
        x = self.entry(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.trans1(x)
        
        x = self.block2(x)
        x = self.pool2(x)
        x = self.trans2(x)
        
        x = self.block3(x)
        
        x = self.gap(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.prm = prm
        self.lr = prm['lr']
        self.batch_size = prm['batch']
        self.transform = prm['transform']
        
        # Use SGD with momentum (standard for CIFAR-10 stability in 1 epoch)
        # Weight decay is small to prevent overfitting in 1 epoch
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=self.lr, 
            momentum=prm['momentum'], 
            weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Calculate steps for cosine decay
        # Assuming standard CIFAR-10 dataset size of 50,000
        total_samples = 50000
        self.max_steps = total_samples // self.batch_size
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        step = 0
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Cosine learning rate schedule
            # t is current step, max_steps is total
            t = step / max(1, self.max_steps)
            # Cosine annealing: lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t))
            # Here lr_min is 0
            current_lr = 0.5 * self.lr * (1 + math.cos(math.pi * t))
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            step += 1