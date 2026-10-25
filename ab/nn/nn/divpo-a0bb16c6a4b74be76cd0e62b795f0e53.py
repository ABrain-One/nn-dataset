import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # derive from in_shape, DO NOT hardcode
        num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self.make_layer(32, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def make_layer(self, inplanes, planes, blocks):
        layers = [nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()


bounds = {"flops": 3.2579269939593314e+07, "params": 5.000000000000001e+05}
class Net(nn.Module):
  # The input shape is (3, 32, 32)
  def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(32)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.layer1 = self.make_layer(32, 64, 2)
    self.layer2 = self.make_layer(64, 128, 2)
    self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
    self.device = device

  def make_layer(self, inplanes: int, planes: int, blocks: int) -> nn.Sequential:
    layers = [nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)]
    layers += [nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

  def get_flops(self):
    input = torch.rand(1, *in_shape).to(self.device)
    flops = tuple(torch.profiler.profile(
      on_trace_ready=torch.profiler.schedule(
        wait=1, warmup=1, active=1, repeat=1),
      activities=torch.profiler.TraceActivity.CONVOLUTION)(
        self, input))
    return flops.total_flops / 1e9

  def forward(self, x: torch.Tensor) -> torch.Tensor: return self.classifier(self.avgpool(self.layer2(self.layer1(self.maxpool(self.bn1(self.relu(self.conv1(x))))))).view(x.size(0), -1))
  def learn(self, train_data):
    self.train()
    for inputs, labels in train_data:
      inputs, labels = inputs.to(self.device), labels.to(self.device)
      self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
      self.optimizer.zero_grad()
      outputs = self(inputs)
      loss = nn.CrossEntropyLoss()(outputs, labels)
      loss.backward()
      nn.utils.clip_grad_norm_(self.parameters(), 3)
      self.optimizer.step()
  def train_setup(self, prm):
    self.to(self.device)
    self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

# Expected and actual memory consumption in bytes
mem_expected, mem_actual = torch.rand(32, 32, 32, 32).to(torch.device('cuda')).nbytes, torch.rand(32, 32, 32, 32).to(torch.device('cuda')).nbytes