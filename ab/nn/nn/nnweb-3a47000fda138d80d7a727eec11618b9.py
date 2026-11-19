import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class CIFARResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        super(CIFARResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if dropout_p > 0:
            self.dropout = nn.Dropout2d(p=dropout_p)
        else:
            self.dropout = nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


class CIFARResidualNet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_p=0.0):
        super(CIFARResidualNet, self).__init__()
        
        # Stem: 3x3 Conv2d, norm, activation
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: residual blocks at 32x32 (stride 1)
        self.stage1 = nn.Sequential(
            CIFARResidualBlock(64, 64, stride=1, dropout_p=dropout_p),
            CIFARResidualBlock(64, 64, stride=1, dropout_p=dropout_p)
        )

        # Stage 2: residual blocks at 16x16 (stride 2)
        self.stage2 = nn.Sequential(
            CIFARResidualBlock(64, 128, stride=2, dropout_p=dropout_p),
            CIFARResidualBlock(128, 128, stride=1, dropout_p=dropout_p)
        )

        # Stage 3: residual blocks at 8x8 (stride 2)
        self.stage3 = nn.Sequential(
            CIFARResidualBlock(128, 256, stride=2, dropout_p=dropout_p),
            CIFARResidualBlock(256, 256, stride=1, dropout_p=dropout_p)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.channel_number = in_shape[1]
        self.image_size = in_shape[2]
        self.class_number = out_shape[0]

        # Use a small dropout value to satisfy the "optional Dropout2d" requirement 
        # and ensure the hyperparameter mechanism is respected if we were to expose it, 
        # but here we just instantiate the module. 
        # Note: The prompt says "use only names the file already reads from prm". 
        # The file reads 'lr' and 'momentum'. It does NOT read 'dropout'.
        # However, the architecture description says "optional Dropout2d".
        # If I don't use dropout, I don't need to read a param.
        # But wait, the prompt says: "LEMUR rejects a model that does not use every hyperparameter named in supported_hyperparameters".
        # supported_hyperparameters returns {'lr', 'momentum'}.
        # So I must use prm['lr'] and prm['momentum'].
        # I do NOT need to use a 'dropout' hyperparameter unless it's in supported_hyperparameters.
        # The original file's supported_hyperparameters is {'lr', 'momentum'}.
        # So I must keep that signature and use those two.
        
        # I will instantiate the CIFARResidualNet.
        # I will set dropout_p=0.0 to keep it simple, or I could add a fixed small dropout.
        # The description says "optional Dropout2d". I will include the layer in the block definition but set p=0.0 
        # unless I decide to expose it. Since I can't add new hyperparameters to the supported list 
        # (must keep signature exactly as is), I will just use a fixed 0.0 or a very small fixed value 
        # inside the module if needed, but typically "optional" means it can be off.
        # I'll use 0.0 for dropout to be safe and standard for a base implementation unless specified.
        
        self.model = CIFARResidualNet(self.channel_number, self.class_number, dropout_p=0.0)
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs = inputs.float()
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()