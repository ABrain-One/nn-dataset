import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.transform = prm.get('transform', 'norm_32')

        # Widths chosen to stay well under 5M params while maintaining depth
        # 96 -> 192 -> 384
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stage 1: 96 channels
        self.stage1 = nn.Sequential(
            FractalBlock(96, 2, 0.0),
            nn.AvgPool2d(2)
        )
        
        # Transition 1: 96 -> 192
        self.trans1 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 192 channels
        self.stage2 = nn.Sequential(
            FractalBlock(192, 2, 0.0),
            nn.AvgPool2d(2)
        )
        
        # Transition 2: 192 -> 384
        self.trans2 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 384 channels
        self.stage3 = nn.Sequential(
            FractalBlock(384, 2, 0.0)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(384, self.num_classes)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        # Use SGD with momentum. Momentum is passed in prm.
        # We use a high initial LR and decay it within the epoch.
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'], 
            weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = prm['batch']
        self.transform = prm['transform']
        # Total steps in one epoch (CIFAR-10 has 50k train samples)
        self.total_steps = 50000 // self.batch_size
        self.current_step = 0
        self.base_lr = prm['lr']

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Linear decay schedule within the single epoch
            # Start at base_lr, end at 0
            progress = self.current_step / self.total_steps
            current_lr = self.base_lr * (1.0 - progress)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Clip gradients for stability with high initial LR
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            self.optimizer.step()
            self.current_step += 1

class FractalBlock(nn.Module):
    def __init__(self, channels, depth, drop_prob):
        super(FractalBlock, self).__init__()
        self.channels = channels
        self.depth = depth
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        if depth > 1:
            self.left = FractalBlock(channels, depth - 1, drop_prob)
            self.right_1 = FractalBlock(channels, depth - 1, drop_prob)
            self.right_2 = FractalBlock(channels, depth - 1, drop_prob)
            self.join = FractalDropPath(drop_prob)
        self.skip = nn.Identity()

    def forward(self, x):
        if self.depth == 1:
            return self.conv(x)
        out_left = self.left(x)
        out_right = self.right_2(self.right_1(x))
        merged = self.join([out_left, out_right])
        # Add residual connection for stability
        return merged + self.skip(x)

class FractalDropPath(nn.Module):
    def __init__(self, drop_prob):
        super(FractalDropPath, self).__init__()
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