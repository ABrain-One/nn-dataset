import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device

        channel_number = in_shape[1]
        class_number = out_shape[0]

        # Stage 1: 3 -> 64 channels
        self.conv1 = nn.Conv2d(channel_number, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Stage 2: 64 -> 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Stage 3: 128 -> 256 channels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Residual-style skip connection in deeper feature stages
        # We implement a simple residual block here for the "deeper" part.
        # Input to this block is 256 channels.
        # To maintain spatial dimensions or handle downsampling, we can use stride 2.
        # However, standard "residual-style" in simple CNNs often implies identity or projection skip.
        # Let's create a residual block that takes 256 in, goes to 256 out (or 512 if we wanted more, but spec says "up to 256").
        # Let's stick to 256 channels max.
        
        # Residual Block 1 (on 256 channels)
        self.res_block1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        
        # Residual Block 2 (on 256 channels)
        self.res_block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=prm['dropout'])
        self.classifier = nn.Linear(256, class_number)

    def forward(self, x):
        # Stage 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Stage 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # Stage 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        
        # Residual connections in deeper stages
        # Identity skip connection
        identity = out
        
        out = self.res_block1(out)
        out += identity
        out = self.relu(out)
        
        # Second residual block
        identity = out
        out = self.res_block2(out)
        out += identity
        out = self.relu(out)
        
        # Global Average Pooling
        out = nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        
        out = self.dropout(out)
        out = self.classifier(out)
        
        return out

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)

        lr = prm['lr']
        momentum = prm['momentum']

        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()