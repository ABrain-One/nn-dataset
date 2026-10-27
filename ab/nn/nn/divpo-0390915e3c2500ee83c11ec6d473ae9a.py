import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_shape[1], out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self.build_layer(in_channels=32, out_channels=64, stride=1)
        self.layer2 = self.build_layer(in_channels=64, out_channels=128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.5))
        self.to(device)

    def build_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.cross_entropy_loss(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def cross_entropy_loss(self, inputs, targets):
        return nn.CrossEntropyLoss()(inputs, targets)

    def train_setup(self, prm):
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
        )

    def __next__(self):
        raise StopIteration
    def __iter__(self):
        yield self

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    def close(self):
        try:
            self.train_setup(prm={})
            self.learn(train_data=iter([([torch.zeros(2, 3, 32, 32), torch.zeros(2, dtype=torch.long)],) for _ in range(10)]))
            self.to('cpu')
        finally:
            self.to('cpu')

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

    def get_complexity(self, input_resolution):
        # Pre-compute for fixed input size
        input_resolution = (input_resolution, input_resolution)
        x = torch.zeros(1, 3, *input_resolution, device=self.device)
        macs, params = get_model_complexity_info(self, (3, *input_resolution), input_resolution=input_resolution, verbose=False)
        return macs, params

    def get_memory(self, input_resolution):
        # Pre-compute for fixed input size
        x = torch.zeros(1, 3, *input_resolution, device=self.device)
        macs, params = get_model_complexity_info(self, (3, *input_resolution), input_resolution=input_resolution, verbose=False)
        flops = macs * 2  # Convert MACs to FLOPs
        peak_mem = (flops / (2 ** 20)) * 4 / 3
        return peak_mem

def get_model_complexity_info(model, input_shape, input_resolution, device, batch_size, verbose):
    # Helper function to get model complexity and memory
    if not torch.cuda.is_available():
        device = 'cpu'
    try:
        model = model.to(device)
        model.eval()
        input_tensor = torch.empty(batch_size, *input_shape, device=device)
        if isinstance(input_resolution, tuple):
            input_tensor = input_tensor.view(batch_size, *input_shape)
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops = macs * 2  # Convert MACs to FLOPs
        peak_mem = (flops / (2 ** 20)) * 4 / 3  # Estimate peak memory
        return macs, params
    finally:
        if device != 'cpu':
            torch.cuda.empty_cache()