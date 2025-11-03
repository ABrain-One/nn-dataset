class Encoder(nn.Module):
       def __init__(self, in_channels, num_classes):
           ...

       def forward(self, x):
           x = self.conv1(x)
           x = self.bn1(x)
           x = self.relu(x)
           x = self.maxpool(x)

           x = self.layer1(x)
           x = self.layer2(x)
           x = self.layer3(x)

           # Now x is [B, 512, 7, 7]
           # Flatten the spatial dimensions
           x = x.permute(0, 2, 3, 1)  # [B, 7, 7, 512]
           x = x.contiguous().view(x.size(0), -1, 512)
           # Project to 768
           x = x.expand(x.size(0), -1, 768)  # But this duplicates the 512 to 768 arbitrarily?

def supported_hyperparameters():
    return {'lr','momentum'}
