import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device

        # Extract necessary values
        in_channels = int(in_shape[1])
        self.vocab_size = int(out_shape[0])

        # Configure encoder backbone (using ResNet-inspired architecture)
        self.hidden_dim = 768  # ≥640, recommended to match decoder expectations

        # Encoder: ReLU layers to extract meaningful image features
        self.encoder = self._build_encoder(in_channels)

        # Decoder: LSTM to generate captions
        self.decoder = self._build_decoder()
        self.linear = nn.Linear(768, self.vocab_size)

    def _build_encoder(self, in_channels):
        # Stem
        encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Stage 1
            self._residual_block(64, 64, 2),
            # Stage 2
            self._residual_block(64, 128, 2),
            # Stage 3
            self._residual_block(128, 256, 2),
            # Stage 4
            self._residual_block(256, 512, 2),
            # Final pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        return encoder

    def _residual_block(self, in_channels, out_channels, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.Sequential(*layers))
        return layers

    def _build_decoder(self):
        # Decoder LSTM
        return nn.LSTM(input_size=512, hidden_size=768, num_layers=1, batch_first=True)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=prm['lr'], **{k: v for k, v in prm.items() if k != 'lr'})

    def learn(self, train_data):
        # train_data is expected to be a tuple (images, captions) on the device
        images, captions = train_data
        features = self.encoder(images)
        features = features.reshape(features.size(0), -1, features.size(1))
        features = features.squeeze(1)
        features = self.linear(features)
        out, _ = self.decoder(features.unsqueeze(0))
        loss = self.criterion(out[0], captions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def forward(self, images, captions):
        # Shape asserts
        assert images.dim() == 4, "Images must be 4D tensors"
        assert captions.dim() == 2, "Captions must be 2D tensors"

        # Encode the images
        features = self.encoder(images)
        # Project to hidden_dim
        features = features.reshape(features.size(0), -1, features.size(1))
        features = features.squeeze(1)
        features = self.linear(features)
        # Run the decoder
        out, _ = self.decoder(features.unsqueeze(0))
        return out

    def supported_hyperparameters():
    return {'lr','momentum'}


# Example usage (not part of the class)
if __name__ == '__main__':
    # This is just to demonstrate that the code is runnable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_shape = (3, 224, 224)  # Example input shape
    out_shape = (10000,)       # Example output shape
    prm = {'lr': 0.001, 'momentum': 0.9}
    model = Net(in_shape, out_shape, prm, device)
    print(model)