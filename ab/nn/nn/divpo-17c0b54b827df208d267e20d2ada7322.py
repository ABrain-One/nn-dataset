import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self.to(device)
        self.criteria = nn.CrossEntropyLoss()
        lr = prm.get('lr', 1e-3)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def supported_hyperparameters():
        return ['lr', 'momentum']
    def supported_prm_values():
        return ['lr', '0.01', 'momentum', '0.9']
    def get_hyper_values(key):
        return {
            'lr': [0.001, 0.003, 0.0003],
            'momentum': [0.8, 0.9, 0.99]
        }.get(key)
    def get_hyper_range(key):
        return {
            'lr': (0.0001, 0.01),
            'momentum': (0.7, 1.0)
        }.get(key)
    def get_best_hyper(metric_key, hyper_key, dataset, model):
        hyper_values = get_hyper_values(hyper_key)
        model.learn(dataset)
        metric_values = []
        for hyper_value in hyper_values:
            model.optimizer.param_groups[0][hyper_key] = hyper_value
            metric_value = eval_model(metric_key, model, dataset)
            metric_values.append(metric_value)
        best_hyper_value = hyper_values[metric_values.index(max(metric_values))]
        return best_hyper_value

def eval_model(metric_key, model, dataset):
    metric_values = []
    for inputs, labels in dataset:
        outputs = model(inputs)
        metric_value = metric_function(metric_key, outputs, labels)
        metric_values.append(metric_value)
    return sum(metric_values) / len(metric_values)

def metric_function(metric_key, outputs, labels):
    if metric_key == 'accuracy':
        return (outputs.argmax(dim=1) == labels).sum().float() / labels.size(0)
    elif metric_key == 'loss':
        return nn.CrossEntropyLoss()(outputs, labels).detach()
    else:
        raise ValueError(f"Unknown metric key: {metric_key}")

class Profile(nn.Module):
    def __init__(self, module, input_shape):
        super().__init__()
        self.module = module
        self.input_shape = input_shape

    def forward(self, x):
        return self.module(x)

    def profile(self, input_size, input_shape):
        self.to('cpu')
        self.eval()
        with torch.no_grad():
            x = torch.randn(*input_shape).to(self.module.device)
            self.module.forward(x)
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            torch.cuda.synchronize()