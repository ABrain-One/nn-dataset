import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class UNet2DModel(nn.Module):
    def __init__(
            self,
            sample_size=64,
            in_channels=1,
            out_channels=256,
            layers_per_block=3,
            block_out_channels=(32, 64, 128),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock3D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock3D"),
    ):
        super(UNet2DModel, self).__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers_per_block = layers_per_block

                                                                                                               
        self.down_blocks = nn.ModuleList()
        in_ch = in_channels
        for out_ch, block_type in zip(block_out_channels, down_block_types):
            if block_type.startswith("Down"):
                self.down_blocks.append(self._make_block(in_ch, out_ch, block_type))
                in_ch = out_ch
            else:
                self.down_blocks.append(self._make_block(in_ch, out_ch, block_type))
                in_ch = out_ch
        
        self.up_blocks = nn.ModuleList()
        for out_ch, block_type in zip(block_out_channels[::-1], up_block_types):
            if out_ch == 384:
                in_ch = 384
            else:
                in_ch = out_ch
            self.up_blocks.append(self._make_block(in_ch, out_ch, block_type))
            in_ch = out_ch

        self.final_conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def _make_block(self, in_channels, out_channels, block_type):
        if block_type.startswith("Down"):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        elif block_type.startswith("Up"):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=2, stride=2
                ),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )

                                                   


class Net(nn.Module):
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        layers = []
        in_channels = in_shape[1]

                                                                                                        
        layers += [
            nn.Conv2d(in_channels, 32, kernel_size=7,
                      stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        in_channels = 32

        layers += [
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        in_channels = 256

        layers += [
            nn.Conv2d(in_channels, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        in_channels = 384

        layers += [
            nn.Conv2d(in_channels, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        in_channels = 384

        layers += [
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        in_channels = 256

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        dropout_p = prm['dropout']
        classifier_input_features = in_channels * 6 * 6
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(classifier_input_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_shape[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

addon_accuracy: 0.982