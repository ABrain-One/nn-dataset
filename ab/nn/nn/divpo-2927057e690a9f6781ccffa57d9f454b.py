import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]  # should be 3 for RGB
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.dropout_prob = prm.get('dropout', 0.3)

        self.model = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((6, 6)),

            nn.Flatten(),

            nn.Linear(128 * 6 * 6, 1024, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_prob),

            nn.Linear(1024, self.num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def get_flops(self, input_size):
        # Estimate FLOPs using torchvision's profile
        dummy_input = torch.randn(input_size, device=self.device)
        total_ops = 0
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    flops, _ = torchvision.utils.flops_counter.get_model_complexity_info(
                        module, input_size, as_str=False)
                    total_ops += flops
        return total_ops

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)