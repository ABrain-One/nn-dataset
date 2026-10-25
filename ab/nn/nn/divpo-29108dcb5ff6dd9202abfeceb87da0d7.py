import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self.build_layer(32, 64, 2, 1)
        self.layer2 = self.build_layer(64, 128, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.5))

    def build_layer(self, in_channels, out_channels, blocks, stride):
        layers = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        ]
        for _ in range(blocks - 1):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)

    def __next_multiple(self, value, multiple_of):
        return int(math.ceil(value / multiple_of)) * multiple_of

    def __calc_size(self, in_size, kernel_size, stride, padding):
        return self.__next_multiple((in_size + padding * 2 - kernel_size) / stride + 1, stride)

    def __conv_output_length(self, size, kernel_size=1, stride=1, padding=0, dilation=1):
        return self.__calc_size(size, kernel_size, stride, padding)

    def __make_same_shape(self, layer, in_channels, out_channels):
        if in_channels != out_channels:
            return nn.Sequential(layer, nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        return layer

    def __apply_conv_shortcut(self, layer, in_channels, out_channels):
        layer = self.__make_same_shape(layer, in_channels, out_channels)
        layer.stride = (1, 1)
        self.__update_layer_input_channels(layer, out_channels)
        return layer

    def __update_layer_input_channels(self, layer, new_channels):
        if isinstance(layer, nn.BatchNorm2d):
            layer.num_features = new_channels
        elif isinstance(layer, nn.Conv2d):
            layer.in_channels = new_channels
        elif isinstance(layer, nn.Linear):
            layer.in_features = new_channels

    def __apply_pool_shortcut(self, layer, in_channels, out_channels):
        self.__update_layer_input_channels(layer, out_channels)
        return layer

    def __apply_shortcut(self, layer, in_channels, out_channels):
        return self.__apply_conv_shortcut(layer, in_channels, out_channels)

    def __apply_pool_or_shortcut(self, layer, in_channels, out_channels, stride):
        if isinstance(layer, nn.AvgPool2d):
            return self.__apply_pool_shortcut(layer, in_channels, out_channels)
        elif isinstance(layer, nn.MaxPool2d):
            return self.__apply_pool_shortcut(layer, in_channels, out_channels)
        elif isinstance(layer, nn.Conv2d):
            return self.__apply_conv_shortcut(layer, in_channels, out_channels)
        elif isinstance(layer, nn.BatchNorm2d):
            return self.__apply_conv_shortcut(layer, in_channels, out_channels)
        elif isinstance(layer, nn.Linear):
            return self.__apply_conv_shortcut(layer, in_channels, out_channels)
        elif isinstance(layer, nn.ReLU):
            return self.__apply_conv_shortcut(layer, in_channels, out_channels)
        else:
            return self.__apply_shortcut(layer, in_channels, out_channels)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    def clear(self):
        for p in self.parameters():
            p.grad = None

    def get_memory_breakpoint(self, in_shape):
        x = torch.rand(in_shape, device=self.device)
        self.train()
        self.to(self.device)
        try:
            self(x)
            return 0, 0, 0
        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise
            return self.get_memory_breakpoint(in_shape)

    def get_memory_breakpoint_edge(self, in_shape):
        x = torch.rand(in_shape, device=self.device)
        self.train()
        self.to(self.device)
        try:
            self(x)
            return 0, 0, 0
        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise
            return self.get_memory_breakpoint_edge(in_shape)

    def get_memory_breakpoint_mobile(self, in_shape):
        x = torch.rand(in_shape, device=self.device)
        self.train()
        self.to(self.device)
        try:
            self(x)
            return 0, 0, 0
        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise
            return self.get_memory_breakpoint_mobile(in_shape)