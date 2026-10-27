import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        if len(in_shape) == 4:  # channels-first (C, H, W) format
            in_channels = in_shape[1]  # derived from (batch, channels, height, width)
        else:  # single channel input (like MNIST)
            in_channels = in_shape[0]
        num_classes = int(out_shape[0])

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.2))
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.max_pool(self.batch_norm1(nn.functional.relu(self.conv1(x))))
        x = self.max_pool(self.batch_norm2(nn.functional.relu(self.conv2(x))))
        x = self.max_pool(self.batch_norm3(nn.functional.relu(self.conv3(x))))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __str__(self):
        return "CIFAR10_Net"

def configure_model(in_shape, out_shape, prm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(in_shape, out_shape, prm, device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count <= 500000, f"Model has {param_count} params, over the limit by {param_count - 500000}"
    return model