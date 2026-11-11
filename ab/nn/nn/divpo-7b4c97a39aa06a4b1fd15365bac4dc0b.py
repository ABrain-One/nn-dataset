import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, expansion=6):
        super().__init__()
        hidden_channels = in_channels * expansion
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, padding=0) if in_channels != out_channels else nn.Identity(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.act(x + self.skip[0](residual))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]   # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=1)
        self.block1 = Block(32, 32, kernel_size=3, padding=1, stride=1)
        self.block2 = Block(32, 64, kernel_size=3, padding=1, stride=2)
        self.block3 = Block(64, 64, kernel_size=3, padding=1, stride=1)
        self.block4 = Block(64, 128, kernel_size=3, padding=1, stride=2)
        self.block5 = Block(128, 128, kernel_size=3, padding=1, stride=1)
        self.block6 = Block(128, 256, kernel_size=3, padding=1, stride=2, expansion=1)
        self.block7 = Block(256, 256, kernel_size=3, padding=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
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

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}


###################### analyze/sanity-check/base-model-complexity/estimate/theoretical-workload ######################
# For a single 32x32 image, estimate macs, params, and activation volume per forward pass.
# For example, a 5x5 conv w/ 32 in/out and stride1 pads1 on 32x32 would:
#   use 5*5*32*32*2=1,310,720 MACs per 32x32 image (5x5 area*32 in/out*2 convs)
#   create 5*5*32*2=16,000 weights per 32x32 image (5*5*32*2 filters)
#   produce 32*32*2=20480 activations per 32x32 image (5*5*32*2*32 area per 32x32 image)
#
# For a layer, add up MACs/params across all 32x32 images.
# For example, the block1/block2[0] (1x1+3x3+1x1) and stem 32->32->64 maps at 32x32:
#   use (1*1+3*3*32*32+1*1)*32=1,048,320 MACs per 32x32 image (32*32 convs, 1x1*32+3x3*32*32*32+1x1*32)
#   create (1*1+3*3*32*2+1*1)*32=100,352 weights (32*32 convs, 1*1*32+3*3*32*32*2+1*1*32)
#   produce 32*32*32*32=1,048,576 activations (32*32*32*32 area)
#
# Use this analysis to calculate theoretical workload:
#   .01*10000/32    = 31.25  MACs/image        (31.25 * 10000 images)
#   .01*10000/32    = 31.25  MACs/param        (31.25 * 10000 weights)
#   .01*10000/32    = 31.25  MACs/activation   (31.25 * 10000 * 32 activations)
# .01*10000/32 + 1  = 32.01  image/param       (32.01 * 10000 weights)
# .01*10000/32 + 1  = 32.01  image/activation  (32.01 * 10000 * 32 activations)
#
#   1024*1024*32    = 33,554,432 images/sec on 1600 MACs/sec GPU
#   1024*1024*32    = 33,554,432 images/sec on 1600 MACs/sec GPU
#   (33.554432 * 1000000) / 1600 = 209716.925  images/sec GPU
#
#   10000*32*32*32  = 33,554,432 params (across 10000 images)
#   10000*32*32*32  = 33,554,432 params (across 10000 images)
#   (33.55443