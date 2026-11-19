import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, drop_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.drop_prob = drop_prob

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        if self.training and self.drop_prob > 0.0:
            # Stochastic Depth
            mask = torch.bernoulli(torch.full((x.size(0), 1, 1, 1), 1 - self.drop_prob, device=x.device))
            out = out * mask
        out += identity
        out = F.relu(out, inplace=True)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        # Stochastic Depth rates (fractal-like randomness)
        # Total 9 blocks (3 per stage)
        self.drop_path_rates = [0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet Blocks
        # Stage 1: 32x32, 64 channels
        self.layer1 = self._make_layer(64, 64, 3, stride=1, start_idx=0)
        # Stage 2: 16x16, 128 channels
        self.layer2 = self._make_layer(64, 128, 3, stride=2, start_idx=3)
        # Stage 3: 8x8, 256 channels
        self.layer3 = self._make_layer(128, 256, 3, stride=2, start_idx=6)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, out_shape[0])
        
        self._initialize_weights()

    def _make_layer(self, in_planes, planes, num_blocks, stride, start_idx):
        downsample = None
        if stride != 1 or in_planes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []
        for i in range(num_blocks):
            # Assign drop path probability based on global block index
            drop_prob = self.drop_path_rates[start_idx + i]
            layers.append(BasicBlock(in_planes if i == 0 else planes, planes, stride if i == 0 else 1, downsample if i == 0 else None, drop_prob))
            # After the first block, in_planes becomes planes for subsequent blocks
            in_planes = planes
            
        return nn.Sequential(*layers)

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
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Use SGD with momentum
        # lr is the base lr
        # We will implement a linear warmup and cosine decay within the epoch
        # CIFAR-10 has 50,000 images.
        # batch size is provided in prm
        batch_size = prm.get('batch', 128)
        total_steps = 50000 // batch_size
        
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'], 
            weight_decay=1e-4
        )
        
        self.total_steps = total_steps
        self.current_step = 0
        self.base_lr = prm['lr']
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            # Adjust learning rate based on step
            # Warmup for first 5% of steps, then cosine decay
            step = self.current_step
            if step < 0.05 * self.total_steps:
                # Linear warmup
                if self.total_steps > 0:
                    lr = self.base_lr * (step / (0.05 * self.total_steps))
                else:
                    lr = self.base_lr
            else:
                # Cosine decay
                if self.total_steps > 0:
                    progress = (step - 0.05 * self.total_steps) / (0.95 * self.total_steps)
                    lr = 0.5 * self.base_lr * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)))
                else:
                    lr = self.base_lr
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            self.current_step += 1