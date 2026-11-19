import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation


class CIFARResidualBlock(nn.Module):
    """
    Residual block for CIFAR-10.
    Uses 3x3 convolutions.
    Includes BatchNorm, ReLU, Dropout2d, SqueezeExcitation, and Stochastic Depth.
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0, use_se=True, use_stochastic_depth=False, stochastic_depth_prob=0.0):
        super(CIFARResidualBlock, self).__init__()
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_prob = stochastic_depth_prob

        # 3x3 Conv -> Norm -> Activation -> Dropout -> SE
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(dropout)

        # 3x3 Conv -> Norm -> Activation -> Dropout -> SE
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(dropout)

        # Squeeze and Excitation
        if use_se:
            self.se = SqueezeExcitation(out_channels, out_channels // 4, activation=nn.SiLU)
        else:
            self.se = nn.Identity()

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.se(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.se(out)

        if self.use_stochastic_depth:
            out = self._stochastic_depth(out)

        out += identity
        out = self.relu2(out) # ReLU again after addition? Standard ResNet usually has ReLU at end of block.
        # Wait, standard ResNet block:
        # y = F(x)
        # out = y + x
        # return ReLU(out)
        # My current structure:
        # out = ... (no relu at end of F(x) path in my code above? 
        # Let's re-verify standard CIFAR ResNet block.
        # Usually: Conv-BN-ReLU, Conv-BN, Add, ReLU.
        # My code: Conv-BN-ReLU-Drop-SE, Conv-BN-ReLU-Drop-SE.
        # Then Add. Then ReLU?
        # If I put ReLU at the end of the main path, then add, then ReLU again, that's double ReLU.
        # Standard: Main path ends with BN. Shortcut ends with BN. Add. ReLU.
        # Let's adjust to standard:
        # Main path: Conv1-BN1-ReLU1, Conv2-BN2.
        # Shortcut: Conv1x1-BN.
        # Add.
        # ReLU.
        
        # Let's rewrite forward to be strictly standard ResNet-CIFAR style but with the extra components.
        # The prompt says: "Blocks use 3x3 Conv2d, norm, activation, optional Dropout2d, optional SE/CBAM, and optional stochastic depth."
        # It doesn't specify exactly where SE and Dropout go. Standard placement is after activation.
        # Stochastic depth is usually applied to the output of the main branch before addition.
        
        # Let's stick to a robust implementation:
        # Path: Conv -> BN -> ReLU -> Dropout -> SE
        # Path: Conv -> BN -> ReLU -> Dropout -> SE
        # This is a bit unusual (SE after ReLU). Usually SE is after BN.
        # Let's assume: Conv -> BN -> ReLU -> Dropout -> SE -> Conv -> BN -> ReLU -> Dropout -> SE
        # Then Add. Then ReLU.
        
        # Actually, looking at the previous code, it used GELU. CIFAR ResNets typically use ReLU.
        # I will use ReLU.
        
        # Let's refine the forward pass based on standard ResNet structure with these additions.
        # 1. Conv1, BN1, ReLU1, Dropout1, SE1
        # 2. Conv2, BN2, Dropout2, SE2 (No ReLU after BN2 in main path? Or ReLU then Add then ReLU?)
        # Standard: Conv1-BN1-ReLU1, Conv2-BN2, Add, ReLU.
        # If I add Dropout and SE, where?
        # Let's put Dropout and SE after the ReLU in the first layer, and after BN in the second layer?
        # Or just after each Conv-BN-ReLU block.
        
        # Let's do:
        # y = x
        # y = Conv1(y); y = BN1(y); y = ReLU1(y); y = Dropout1(y); y = SE1(y)
        # y = Conv2(y); y = BN2(y); y = Dropout2(y); y = SE2(y)
        # y = y + identity
        # y = ReLU(y)
        
        # This seems reasonable.
        
        # Wait, I already defined the layers in __init__. I need to make sure forward matches.
        # In __init__:
        # self.conv1, self.bn1, self.relu1, self.dropout1
        # self.conv2, self.bn2, self.relu2, self.dropout2
        # self.se (applied to out_channels)
        
        # If I apply SE after ReLU2, that's fine.
        # Let's assume the "activation" in the description implies the standard ReLU.
        
        # Corrected Forward:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.se(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # Note: Standard ResNet doesn't have ReLU after BN2 before add.
        # But the prompt says "Blocks use 3x3 Conv2d, norm, activation...".
        # If I strictly follow "Conv, norm, activation", then:
        out = self.relu2(out) # Activation after norm
        out = self.dropout2(out)
        out = self.se(out)

        if self.use_stochastic_depth:
            out = self._stochastic_depth(out)

        out += identity
        out = self.relu2(out) # Final activation
        
        return out

    def _stochastic_depth(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.stochastic_depth_prob
        mask = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) + keep_prob) > 1.0
        mask = mask.float()
        mask = mask / keep_prob
        return x * mask


class CIFARResidualNet(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(CIFARResidualNet, self).__init__()
        self.device = device
        
        # Hyperparameters
        self.lr = prm['lr']
        self.momentum = prm['momentum']
        self.dropout = prm['dropout']
        self.attention_dropout = prm['attention_dropout']
        self.stochastic_depth_prob = prm['stochastic_depth_prob']
        
        num_classes = out_shape[0]
        in_channels = in_shape[1]
        
        # Architecture parameters
        # CIFAR-10 is 32x32.
        # Stem: 3x3 Conv, no padding? Or padding 1?
        # Standard CIFAR ResNet stem: Conv 3x3, stride 1, padding 1. Output 32x32.
        # Stage 1: 32x32
        # Stage 2: 16x16 (stride 2)
        # Stage 3: 8x8 (stride 2)
        
        # Widths: The prompt doesn't specify widths, but "custom small residual network".
        # Typical CIFAR ResNet widths: 64, 128, 256.
        # Let's use [64, 128, 256] as standard.
        widths = [64, 128, 256]
        depths = [2, 2, 2] # Standard depth 6 (2+2+2) or similar. Let's use 2 blocks per stage.
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: 32x32
        self.stage1 = self._make_stage(widths[0], widths[0], depths[0], stride=1, dropout=self.dropout, use_se=True, use_stochastic_depth=False)
        
        # Stage 2: 16x16
        self.stage2 = self._make_stage(widths[0], widths[1], depths[1], stride=2, dropout=self.dropout, use_se=True, use_stochastic_depth=False)
        
        # Stage 3: 8x8
        # Stochastic depth is often applied in deeper stages.
        self.stage3 = self._make_stage(widths[1], widths[2], depths[2], stride=2, dropout=self.dropout, use_se=True, use_stochastic_depth=True)
        
        # Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(widths[2], num_classes)
        
        # Initialize weights
        self._init_weights()
        
        # Note: We must use attention_dropout somewhere.
        # Since this is a ConvNet, "attention_dropout" is a bit weird.
        # However, the rules say "each of those names... must still be read from prm and actually used".
        # If I don't have attention layers, I can't use attention_dropout in an attention layer.
        # I can use it as a dropout in the classifier or somewhere else?
        # Or maybe the "SE" module uses a dropout?
        # SqueezeExcitation in torchvision doesn't have a dropout parameter by default.
        # I can add a dropout layer after SE or in the classifier.
        # Let's add a dropout layer before the classifier using attention_dropout.
        self.classifier_dropout = nn.Dropout(self.attention_dropout)
        
    def _make_stage(self, in_channels, out_channels, depth, stride, dropout, use_se, use_stochastic_depth):
        layers = []
        # First block has stride
        layers.append(CIFARResidualBlock(in_channels, out_channels, stride=stride, dropout=dropout, use_se=use_se, use_stochastic_depth=use_stochastic_depth, stochastic_depth_prob=self.stochastic_depth_prob if use_stochastic_depth else 0.0))
        for _ in range(depth - 1):
            layers.append(CIFARResidualBlock(out_channels, out_channels, stride=1, dropout=dropout, use_se=use_se, use_stochastic_depth=use_stochastic_depth, stochastic_depth_prob=self.stochastic_depth_prob if use_stochastic_depth else 0.0))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.classifier_dropout(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'stochastic_depth_prob'}


class Net(CIFARResidualNet):
    pass