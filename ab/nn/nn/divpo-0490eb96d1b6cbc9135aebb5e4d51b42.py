import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        if len(in_shape) == 4:
            in_channels = in_shape[1]
        else:
            in_channels = int(in_shape[0])
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.stem.weight.data.normal_(0, 0.1)
        self._create_blocks(32)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(32 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def _create_blocks(self, in_channels):
        layers = []
        layers += [
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
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

    def __next_block_id(self):
        for i in range(1000):
            if not f'block{i}' in self.__dict__:
                return i
        raise Exception('Too many blocks')

    # BLOCK DESIGN GOES HERE
    def _design(self):
        # Use self._next_block_id to get a unique id and define Block below
        # Example for Block class (simplest variant):
        # class Block(nn.Module):
        #     def __init__(self, in_channels):
        #         super().__init__()
        #         self.block = nn.Sequential(
        #             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
        #             nn.BatchNorm2d(in_channels),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
        #             nn.BatchNorm2d(in_channels)
        #         )

        id = self.__next_block_id()
        block_id = 'block' + str(id)
                            
        class Block(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels)
                )

            def forward(self, x):
                return self.block(x)

        self.__setattr__(block_id, Block(32))
        self._blocks.append(self.__getattr__(block_id))

    _blocks = []
    _design = _design