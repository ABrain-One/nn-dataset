import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'dropout', 'lr', 'momentum'}

def transform(norm):
    return transforms.Compose([transforms.Resize((256, 256)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(degrees=15), transforms.CenterCrop((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])

def depthwise_separable_conv(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

class InvertedResidual(nn.Module):

    def __init__(self, in_channels, expansion_factor, out_channels, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expanded_dim = expansion_factor * in_channels
        self.conv = nn.Sequential(depthwise_separable_conv(in_channels, self.expanded_dim, 1), depthwise_separable_conv(self.expanded_dim, out_channels, stride))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class SqueezeExcitation(nn.Module):

    def __init__(self, channels, reduction_ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        se_weights = self.se(x)
        return x * se_weights

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.use_amp = prm.get('use_amp', False)
        dropout_prob = float(prm['dropout'])
        self.in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.features = nn.Sequential(nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), InvertedResidual(64, 2, 128, 2), SqueezeExcitation(128), InvertedResidual(128, 2, 256, 2), SqueezeExcitation(256), InvertedResidual(256, 1, 512, 1), SqueezeExcitation(512), InvertedResidual(512, 1, 512, 1), SqueezeExcitation(512))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(p=dropout_prob), nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(p=dropout_prob), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor, is_probing: bool=False) -> torch.Tensor:
        x = x.to(self.device)
        x = self.features(x)
        if is_probing:
            return x
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device).float()
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def predict(self, test_data):
        self.eval()
        with torch.no_grad():
            for inputs, labels in test_data:
                inputs = inputs.to(self.device).float()
                outputs = self(inputs)
                _, preds = torch.max(outputs, dim=1)
                return (preds, labels)

    def predict_batch(self, batch_size, test_data):
        self.eval()
        with torch.no_grad():
            for inputs, labels in test_data:
                inputs = inputs.to(self.device).float()
                outputs = self(inputs)
                _, preds = torch.max(outputs, dim=1)
                return (preds, labels)