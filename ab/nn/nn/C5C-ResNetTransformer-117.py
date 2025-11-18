class EncoderCNN(nn.Module):
    def __init__(self, encoded_dim=768):
        super(EncoderCNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=4, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        
        # Calculate the actual output channels to be at least 640?
        # Instead, we can compute the output shape and then adjust the final linear projection.
        # Since we cannot predefine too much, let's assume we use enough layers to get 512 and then project to 768.

        self.dropout = nn.Dropout()
        self.fc = nn.Linear(512 * 7 * 7, encoded_dim)   # After 7th convolution, the image is reduced by factor 128 (original 224/128) and then flatten? 

        # Wait, the original input size isn't defined, but let's assume square image of side length divisible by 4 (so that maxpool doesn't leave fractional parts).

        # Alternatively, we can use global average pooling at every layer and then concatenate?

        # Let me change my mind: I want the feature map to be 7x7 at the deepest layer, so the output of the last convolution should be 512, and then we can flatten to 512*7*7 and project to 768.

        # But 512*7*7 is very big (around 250k parameters) and might be heavy. Alternatively, we can use global average pooling again.

    def forward(self, image):
        # image: [batch, channel, height, width]
        # Convolution layers
        x = self.relu(self.conv1(image))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = self.relu(self.conv6(x))
        x = self.pool(x)
        x = self.relu(self.conv7(x))
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, output_size=(1,1)).flatten()
        x = self.dropout(x)
        x = self.fc(x)
        x = x.unsqueeze(1)   # [batch, 1, 768]
        return x

def supported_hyperparameters():
    return {'lr','momentum'}
