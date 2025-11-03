import torch
import torch.nn as nn

class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
            
        out += identity
        out = self.relu(out)
        return out

class ResNetSpatialEncoder(torch.nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        
        # Define the stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Define the rest of the ResNet
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 23, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection to hidden_size
        self.fc = nn.Linear(512 * 7 * 7, out_shape[0])

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        _layer_list = []
        for i in range(blocks):
            _layer_list.append(BasicBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*_layer_list)

    def forward(self, images):
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SpatialAttentionLSTMDecoder(torch.nn.Module):
    def __init__(self, vocab_size, prm, device):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = prm['hidden_size']
        
        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, vocab_size)

    def init_zero_hidden(self, batch_size):
        # Initialize the hidden state and cell state to zeros
        return (torch.zeros(1, batch_size, self.hidden_size).to(self.device),
                torch.zeros(1, batch_size, self.hidden_size).to(self.device))

    def forward(self, inputs, hidden_state=None, features=None):
        # If hidden_state is None, initialize it
        if hidden_state is None:
            hidden_state = self.init_zero_hidden(inputs.size(0))
            
        # Embed the inputs
        embedded = self.embedding(inputs)
        # If features are provided, use them for attention
        if features is not None:
            # features: [B, num_regions, hidden_size]
            # embedded: [B, T, hidden_size]
            embedded = embedded.permute(0, 2, 1)
            features = features.permute(0, 2, 1)
            # Apply attention
            context, attn_weights = nn.functional.multi_head_attention_forward(
                query=embedded,
                key=features,
                value=features,
                embed_dim=embedded.size(1),
                num_heads=8,
                dropout=0.0,
                return_weights=True
            )
            context = context.permute(0, 2, 1)
            # Concatenate context and embedded
            output = torch.cat((context, embedded), dim=2)
            output, hidden_state = self.lstm(output, hidden_state)
            output = self.fc(output)
        else:
            # Just use the embedded input
            output, hidden_state = self.lstm(embedded, hidden_state)
            output = self.fc(output)
        return output, hidden_state

class Net(torch.nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        
            # ---- API aliases (auto-injected) ----

            self.in_shape = in_shape
            self.out_shape = out_shape
            self.device = device
            self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3
            self.vocab_size = out_shape[0] if isinstance(out_shape, (tuple, list)) else int(out_shape)
            self.out_dim = self.vocab_size
            self.num_classes = self.vocab_size
            # Backward-compat local aliases (old LLM patterns)
            vocab_size = self.vocab_size
            out_dim = self.vocab_size
            num_classes = self.vocab_size
            in_channels = self.in_channels
super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.prm = prm
        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3
        self.vocab_size = out_shape[0][0] if isinstance(out_shape, (tuple, list)) else int(out_shape)
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size
        
        # Backward-compat local aliases (old LLM patterns)
        self.encoder = ResNetSpatialEncoder(in_shape, self.vocab_size, self.prm, self.device)
        self.decoder = SpatialAttentionLSTMDecoder(self.vocab_size, self.prm, self.device)
        self.criterion = None
        self.optimizer = None

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999)
        )


    def learn(self, train_data):
        # Train the model on the given data
        self.train()
        for epoch in range(self.prm['epochs']):
            for batch in train_data:
                images, captions = batch
                images = images.to(self.device)
                captions = captions.to(self.device)

                # Forward pass
                outputs, hidden_state = self(images, captions)

                # Compute loss
                loss = self.criterion(outputs.view(-1, self.vocab_size), captions.view(-1))

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def forward(self, images, captions=None, hidden_state=None):
        # If captions are provided, do teacher forcing
        if captions is not None:
            # Get the encoder features
            encoder_features = self.encoder(images)
            # Get the decoder output
            output, hidden_state = self.decoder(captions, hidden_state, encoder_features)
            return output, hidden_state

        # Otherwise, generate captions
        self.eval()
        with torch.no_grad():
            # Start with the start-of-sentence token
            caption = torch.tensor([[self.decoder.sos_index]], device=self.device)
            captions = []
            for i in range(self.prm['max_length']):
                encoder_features = self.encoder(images)
                output, hidden_state = self.decoder(caption, hidden_state, encoder_features)
                caption = torch.argmax(output, dim=1)
                captions.append(caption)
            captions = torch.cat(captions, dim=0)
            return captions, hidden_state

def supported_hyperparameters():
    return {'lr','momentum'}