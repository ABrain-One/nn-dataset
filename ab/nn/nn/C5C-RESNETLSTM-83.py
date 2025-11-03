import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer

def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__) -> None:
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.vocab_size = out_shape[0][0] if isinstance(out_shape, (tuple, list)) else int(out_shape)
        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size
        self.device = device
        self.prm = prm
        
        # Encoder: ResNet50 backbone with modifications
        self.cnn_encoder = self._build_encoder()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, self.vocab_size)
        
        # Decoder: Transformer-based
        self.rnn = self._build_decoder()
        
    def _build_encoder(self):
        # Modified ResNet50 backbone
        _layer_list = [2, 2, 2, 2]
        in_channels = self.in_channels
        out_channels = 64
        model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(out_channels, 64, layers[0]),
            self._make_layer(64, 128, layers[1], stride=2),
            self._make_layer(128, 256, layers[2], stride=2),
            self._make_layer(256, 512, layers[3], stride=2),
        )
        return model

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        # Bottle neck layer
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        _layer_list = []
        _layer_list.append(Bottleneck(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            _layer_list.append(Bottleneck(out_channels * 4, out_channels * 4))
        return nn.Sequential(*_layer_list)

    def _build_decoder(self):
        # Transformer decoder
        d_model = self.vocab_size
        num_heads = 8
        num_layers = 1
        decoder_layer = TransformerDecoderLayer(d_model, num_heads)
        decoder = TransformerDecoder(decoder_layer, num_layers)
        return decoder

    def forward(self, images, captions):
        # images: [B, C, H, W]
        # captions: [B, T]
        # Encoder forward pass
        features = self.cnn_encoder(images)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        
        # Decoder forward pass
        # Convert captions to embeddings
        embedded = F.embedding(captions, self.rnn.embedding)
        embedded = embedded.permute(1, 0, 2)  # [T, B, d_model]
        output = self.rnn(embedded, memory=features)
        output = output.permute(1, 0, 2)  # [B, T, d_model]
        
        # Final projection to vocabulary size
        logits = self.rnn.fc(output)
        return logits

    def init_zero_hidden(self, batch: int, device: torch.device) -> (torch.Tensor, torch.Tensor):
        # For transformer decoder, hidden state is not used
        return (torch.zeros(batch, self.vocab_size, device=device), torch.zeros(batch, self.vocab_size, device=device))

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999)
        )


    def learn(self, train_data):
        # Process training data
        pass

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# Example usage (not part of the model)
if __name__ == '__main__':
    # Create a sample model
    in_shape = (3, 224, 224)
    out_shape = 768
    prm = {'lr': 0.001, 'momentum': 0.9, 'dropout': 0.1}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(in_shape, out_shape, prm, device)
    print("Model created successfully.")