import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Net(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768):
        super(Net, self).__init__()
        self.encoder = self.build_encoder(hidden_dim)
        self.decoder = self.build_decoder(vocab_size, hidden_dim)
        
    def build_encoder(self, hidden_dim):
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, hidden_dim, kernel_size=1)
        )
    
    def build_decoder(self, vocab_size, hidden_dim):
        return nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
    
    def train_setup(self, hparams):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=hparams.lr)
        self.vocab_size = vocab_size
        
    def learn(self, images, captions):
        # images: [B, C, H, W]
        # captions: [B, T]
        inputs = captions[:, :-1]
        targets = captions[:, 1:]
        memory = self.encoder(images)
        embedded = self.decoder_embedding(inputs)
        output = self.decoder(embedded, memory)
        logits = self.decoder_final(output)
        loss = self.criterion(logits.view(-1, self.vocab_size), targets.view(-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def forward(self, images, captions):
        # images: [B, C, H, W]
        # captions: [B, T]
        inputs = captions[:, :-1]
        targets = captions[:, 1:]
        memory = self.encoder(images)
        embedded = self.decoder_embedding(inputs)
        output = self.decoder(embedded, memory)
        logits = self.decoder_final(output)
        return logits, memory

class ImageCaptionDataset(Dataset):
    def __init__(self, images, captions):
        self.images = images
        self.captions = captions
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        return self.images[idx], self.captions[idx]

def supported_hyperparameters():
    return {'lr','momentum'}


# Example usage
if __name__ == '__main__':
    # Create dummy data
    images = torch.randn(32, 3, 224, 224)
    captions = torch.randint(0, 10000, (32, 30))
    
    # Create dataset and dataloader
    dataset = ImageCaptionDataset(images, captions)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    net = Net(vocab_size=10000)
    
    # Train setup
    hparams = supported_hyperparameters()
    net.train_setup(hparams)
    
    # Training loop
    for images_batch, captions_batch in dataloader:
        loss = net.learn(images_batch, captions_batch)
        print(f'Loss: {loss}')