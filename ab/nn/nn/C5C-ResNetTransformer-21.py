class CNNEncoder(nn.Module):
       def __init__(self, input_channels=3, output_dim=768):
           super(CNNEncoder, self).__init__()
           # We'll set output_dim = 768
           # Stage1: 224 -> 56, channels=3*4 (since we use two convolutions: [3,64] -> then [64,128] etc.)
           self.stem = nn.Sequential(
               nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
               nn.BatchNorm2d(64),
               nn.ReLU(inplace=True),
               nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
           )
           self.layer1 = self._make_layer(64, 64, 2)   # 256? We want to increase the channels eventually.
           self.layer2 = self._make_layer(64, 128, 3, stride=2)
           self.layer3 = self._make_layer(128, 256, 4, stride=2)
           self.layer4 = self._make_layer(256, 512, 6, stride=2)

           self.global_pool = nn.AdaptiveAvgPool2d(1)
           self.fc = nn.Linear(512, output_dim)

       def _make_layer(self, in_channel, out_channel, blocks, stride=1):
           # We'll use bottle neck for faster convergence
           layers = []
           for i in range(blocks):
               if i == 0:
                   layers.append(Bottleneck(in_channel, out_channel, stride))
               else:
                   layers.append(Bottleneck(out_channel, out_channel))
           return nn.Sequential(*layers)

       # But note: we don't have the Bottleneck class here. We can borrow from the fifth example or define a minimal one.

       # Minimal Bottleneck block:

       class Bottleneck(nn.Module):
           expansion = 4

           def __init__(self, inplanes, planes, stride=1, downsample=None):
               super(Bottleneck, self).__init__()
               self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
               self.bn1 = nn.BatchNorm2d(planes)
               self.relu = nn.ReLU(inplace=True)
               # Only conv3 changes if the downsample is applied? We won't worry about groups and dilation.
               self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
               self.bn2 = nn.BatchNorm2d(planes)
               self.relu = nn.ReLU(inplace=True)
               self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
               self.bn3 = nn.BatchNorm2d(planes * self.expansion)
               self.downsample = downsample
               # We'll set the in/out dimensions correctly

           def forward(self, x):
               identity = x

               out = self.conv1(x)
               out = self.bn1(out)
               out = self.relu(out)

               out = self.conv2(out)
               out = self.bn2(out)
               out = self.relu(out)

               out = self.conv3(out)
               out = self.bn3(out)

               if self.downsample is not None:
                   identity = self.downsample(identity)

               out += identity
               out = self.relu(out)

               return out

def supported_hyperparameters():
    return {'lr','momentum'}
