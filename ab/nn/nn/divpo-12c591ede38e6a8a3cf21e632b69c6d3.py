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
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(32, 32, 2, dropout=prm.get('dropout', 0.0))
        self.layer2 = self._make_layer(32, 64, 2, dropout=prm.get('dropout', 0.0))
        self.layer3 = self._make_layer(64, 128, 2, dropout=prm.get('dropout', 0.0))
        self.layer4 = self._make_layer(128, 256, 2, dropout=prm.get('dropout', 0.0))
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, out_shape[0])
        self._last_layer_name = 'classifier'  # for export
        self._last_dim = out_shape[0]
        self._dropout = nn.Dropout(p=prm.get('dropout', 0.0)) if prm.get('dropout', 0.0) > 0 else None
        self._size = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self.to(device)

    def _make_layer(self, in_planes, out_planes, blocks, stride=1, dropout=0.0):
        layers = []
        layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for _ in range(blocks - 1):
            layers.append(nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_planes))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0005)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        if self._dropout is not None:
            x = self._dropout(x)
        return self.classifier(x)

    def export_to_onnx(self, filename):
        # Dummy input for ONNX export
        x = torch.randn(1, 3, 32, 32)
        self.eval()
        # Use torch.no_grad() to avoid tracking history in ONNX export
        with torch.no_grad():
            output = self(x)
            # Check if we're exporting the last layer
            if self._last_layer_name == 'classifier':
                # If so, we need to export the logits
                output = output.argmax(dim=1)
            # Export the model
            torch.onnx.export(self,               # model being run
                              x,                         # model input (or a tuple for multiple inputs)
                              filename,   # where to save the model (can be a file or file-like object)
                              export_params=True,        # store the trained parameter weights inside the model file
                              opset_version=10,          # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['input'],   # the model's input names
                              output_names=[self._last_layer_name], # the model's output names
                              dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                            self._last_layer_name: {0: 'batch_size'}})