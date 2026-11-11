import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class SmallConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        return nn.functional.relu(out)

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.width = in_shape[2]
        self.height = in_shape[3]
        self.out_shape = out_shape
        self.num_classes = out_shape[0]
        self.current_channels = 16
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        
        self.features_extractor = self.build_features_extractor()
        self.classifier = nn.Linear(self.current_channels, self.num_classes)
        
    def build_features_extractor(self):
        layers = []
        in_channels = self.in_channels
        for stage, (channels, repeats, stride) in enumerate(self._feature_extractor_stages()):
            for i in range(repeats):
                stride = stride if i == 0 else 1
                layers += self._build_stage_block(channels, stride, in_channels, block_type='small')
                in_channels = channels
        return nn.Sequential(*layers)
    
    def _feature_extractor_stages(self):
        return [
            (16, 3, 1),
            (32, 3, 2),
            (64, 3, 2)
        ]
    
    def _build_stage_block(self, out_channels, stride, in_channels, block_type):
        blocks = []
        if block_type == 'small':
            block = SmallConvBlock
        blocks.append(block(in_channels, out_channels, stride))
        self.current_channels = out_channels
        return blocks
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features_extractor(x)
        x = self._global_average_pooling(x)
        return self.classifier(x)
    
    def _global_average_pooling(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.adaptive_avg_pool2d(x, 1).flatten(start_dim=1)
    
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