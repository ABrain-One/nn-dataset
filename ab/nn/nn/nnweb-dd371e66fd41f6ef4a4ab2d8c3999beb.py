import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum', 'transform'}

class FractalBlock(nn.Module):
    """
    Recursive fractal depthwise-separable block with stochastic depth.
    Structure:
      x -> [left: FractalBlock(n-1)] and [right: FractalBlock(n-1) -> FractalBlock(n-1)]
      -> stochastic drop/mean -> depthwise conv -> pointwise conv -> BN -> GELU
    """
    def __init__(self, n_columns: int, channels: int, drop_prob: float):
        super().__init__()
        self.n_columns = int(n_columns)
        channels = int(channels)
        
        # Depthwise + Pointwise (Separable)
        self.conv_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.conv_pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        
        if self.n_columns > 1:
            self.left = FractalBlock(self.n_columns - 1, channels, drop_prob)
            self.right_1 = FractalBlock(self.n_columns - 1, channels, drop_prob)
            self.right_2 = FractalBlock(self.n_columns - 1, channels, drop_prob)
            self.drop_prob = drop_prob
        else:
            self.drop_prob = drop_prob

    def forward(self, x):
        if self.n_columns == 1:
            # Leaf node: just the separable conv block
            out = self.conv_dw(x)
            out = self.conv_pw(out)
            out = self.bn(out)
            out = self.act(out)
            return out
        
        out_left = self.left(x)
        out_right = self.right_2(self.right_1(x))
        
        # Stochastic Depth (FractalDropPath logic)
        if self.training:
            n = 2
            mask = torch.bernoulli(torch.full((n,), 1 - self.drop_prob, device=x.device))
            if mask.sum() == 0:
                mask[torch.randint(0, n, (1,)).item()] = 1.0
            active = [out_left if mask[0] > 0 else None, out_right if mask[1] > 0 else None]
            active = [a for a in active if a is not None]
            out = torch.stack(active).mean(dim=0)
        else:
            out = (out_left + out_right) / 2.0
            
        # Apply the block's own convs
        out = self.conv_dw(out)
        out = self.conv_pw(out)
        out = self.bn(out)
        out = self.act(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        c_in = in_shape[1] # usually 3
        n_classes = out_shape[0] if out_shape else 10
        start_chan = 64
        
        # Entry: 3x3 conv, BN, GELU
        self.entry = nn.Sequential(
            nn.Conv2d(c_in, start_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(start_chan),
            nn.GELU()
        )
        
        blocks = []
        pools = []
        trans_layers = []
        cur_chan = start_chan
        total_blocks = 3
        
        # Fractal depth 3 (from Model 1)
        fractal_depth = 3
        
        for i in range(total_blocks):
            # Drop prob increases slightly with depth
            drop_prob = 0.1 * (i + 1)
            blocks.append(FractalBlock(fractal_depth, cur_chan, drop_prob))
            pools.append(nn.AvgPool2d(2))
            
            if i < total_blocks - 1:
                next_chan = int(cur_chan * 2) 
                # Transition: 1x1 conv to change channels
                trans_layers.append(nn.Sequential(
                    nn.Conv2d(cur_chan, next_chan, kernel_size=1, bias=False),
                    nn.BatchNorm2d(next_chan),
                    nn.GELU()
                ))
                cur_chan = next_chan
            else:
                trans_layers.append(None)

        self.blocks = nn.ModuleList(blocks)
        self.pools = nn.ModuleList(pools)
        self.trans_layers = nn.ModuleList([t for t in trans_layers if t is not None])
        self.final_channels = cur_chan
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        # Using a small MLP head with GELU
        self.fc1 = nn.Linear(self.final_channels, self.final_channels * 2)
        self.fc_bn = nn.BatchNorm1d(self.final_channels * 2)
        self.fc_act = nn.GELU()
        self.fc_drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(self.final_channels * 2, n_classes)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Normalize input if transform is norm_32
        if self.transform == 'norm_32' or self.transform == 'norm_32_flip':
            # Standard CIFAR-10 normalization
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(x.device)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(x.device)
            x = (x - mean) / std
            
        x = self.entry(x)
        t_idx = 0
        for i, (block, pool) in enumerate(zip(self.blocks, self.pools)):
            x = block(x)
            x = pool(x)
            if i < len(self.trans_layers):
                x = self.trans_layers[t_idx](x)
                t_idx += 1
        
        x = self.global_pool(x)
        x = x.flatten(1)
        
        x = self.fc1(x)
        x = self.fc_bn(x)
        x = self.fc_act(x)
        x = self.fc_drop(x)
        x = self.fc2(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # SGD with Momentum (from Model 3/4 style, but tuned for 1 epoch)
        # Model 1 used AdamW, Model 3/4 used SGD. 
        # For 1 epoch, SGD with high momentum and warmup often generalizes better or learns faster 
        # if lr is tuned, but AdamW is robust. 
        # Let's use SGD with Nesterov-like momentum (standard PyTorch momentum) and a cosine schedule 
        # implemented manually in learn, or just a fixed LR. 
        # Given the constraint of 1 epoch, a warmup + decay schedule is crucial.
        
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'], 
            weight_decay=1e-4
        )
        
        # Store total steps for scheduling
        # We don't know the exact batch count yet, but we can estimate or just use a fixed schedule based on steps.
        # Since learn iterates over the loader, we can track steps.
        self.total_steps = 0 # Will be updated in learn
        self.current_step = 0
        self.warmup_steps = 500 # ~10% of 5000 steps (CIFAR-10 has 50k images)
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        # Estimate total steps if not provided
        # CIFAR-10 has 50000 training images.
        # If batch size is 128, ~390 steps. If 16, ~3125 steps.
        # We need to know the dataset size to schedule properly.
        # train_data is a DataLoader.
        # We can get the length if it's defined.
        try:
            total_samples = len(train_data.dataset)
            batch_size = train_data.batch_size
            self.total_steps = total_samples // batch_size
        except:
            self.total_steps = 5000 # Fallback

        self.current_step = 0
        # Store base lr and momentum from the optimizer's param groups to avoid relying on external 'prm'
        base_lr = self.optimizer.param_groups[0]['lr']
        
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Learning Rate Schedule: Cosine with Warmup
            # Since we only have 1 epoch, we want to start high and decay.
            # However, a simple constant or linear decay is safer if we don't know exact steps.
            # Let's implement a simple cosine decay from 0 to max_lr over warmup, then decay to min_lr.
            
            # Base LR is base_lr (derived from optimizer)
            
            # Warmup phase
            if self.current_step < self.warmup_steps:
                lr_scale = self.current_step / max(1, self.warmup_steps)
            else:
                # Cosine decay
                progress = (self.current_step - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
                # Clamp progress to [0, 1] to handle edge cases
                progress = min(1.0, max(0.0, progress))
                lr_scale = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
                
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = base_lr * lr_scale

            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.current_step += 1