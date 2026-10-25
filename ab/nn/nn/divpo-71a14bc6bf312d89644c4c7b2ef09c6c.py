import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Ensure this aligns with the expected input channels for the dataset
        num_classes = int(out_shape[0])

        self.stem = nn.Sequential(
            Block(in_channels, 32, stride=1, padding=1),
            Block(32, 64, stride=2),
            Block(64, 128, stride=2)
        )

        self.layer1 = Block(128, 128)
        self.layer2 = Block(128, 256, stride=2)
        self.layer3 = Block(256, 256)
        self.layer4 = Block(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.layer2(x)
        x = self.layer3(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __repr__(self):
        return f"Net(in_shape={self.in_shape}, out_shape={self.out_shape}, prm={self.prm}, device={self.device})"

    def __str__(self):
        return self.__repr__()

def train_setup(model, prm):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train_setup(prm)
    model.device = device
    return model

def learn(model, train_data):
    model.learn(train_data)
    return model

def net_id():
    return "_____[Net ID]_:[edge-friendly]_1001"

def name():
    return "edge_friendly_1001"

def academic_integrity():
    return "This model is designed based on the CIFAR-10 dataset and is optimized for fast inference to meet edge device constraints."

def metadata():
    return {
        "name": name(),
        "model_size": 1001,
        "training_details": {
            "dataset": "CIFAR-10",
            "epochs": 1,
            "learning_rate": 0.01,
            "momentum": 0.9
        },
        "academic_integrity": academic_integrity(),
        "hyperparameters": {
            "lr": 0.01,
            "momentum": 0.9
        },
        "edge_friendly": True
    }

def supported_hyperparameters():
    return {'lr', 'momentum'}

def get_model(in_shape, out_shape, prm):
    model = Net(in_shape, out_shape, prm, device=torch.device('cpu'))
    model._metadata = metadata()
    return model

# Dummy entry point to satisfy requirements (not executed)
def main(args):
    print("This is the entry point for the CIFAR-10 model, but it does nothing.")

if __name__ == "__main__":
    class DummyArgs:
        def __init__(self):
            self.model_path = "path/to/model.pth"
    args = DummyArgs()
    main(args)