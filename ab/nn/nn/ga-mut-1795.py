import torch
import torch.nn as nn


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        layers = []
        in_channels = in_shape[1]
        count_included = 0

        # Process all convolutional layers (1-5) but skip excluded layers
        for i in range(1, 6):
            include_key = f'include_conv{i}'
            if prm.get(include_key, 0) == 0:
                continue
                
            count_included += 1
            filters = prm[f'conv{i}_filters']
            kernel = prm[f'conv{i}_kernel']
            
            # Only conv1 has special stride
            stride = prm['conv1_stride'] if i == 1 else 1
            padding = (kernel - 1) // 2  # Maintain spatial size for stride=1
            
            # Add convolutional layer
            layers.append(nn.Conv2d(in_channels, filters, kernel, stride, padding))
            
            # Add batch normalization if enabled
            if prm['use_batchnorm']:
                layers.append(nn.BatchNorm2d(filters))
                
            # Add activation
            layers.append(nn.LeakyReLU(inplace=True))
            
            # Add pooling after 1st, 2nd, or 3rd included layer
            if count_included == 1:
                pooling_type = prm['pooling_type1']
            elif count_included == 2:
                pooling_type = prm['pooling_type2']
            elif count_included == 3:
                pooling_type = prm['pooling_type3']
            else:
                pooling_type = None
                
            if pooling_type is not None:
                # Handle both adaptive and non-adaptive types practically
                if 'Max' in pooling_type:
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                elif 'Avg' in pooling_type:
                    layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            
            in_channels = filters

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        classifier_input_features = in_channels * 6 * 6
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm['dropout']),
            nn.Linear(classifier_input_features, prm['fc1_neurons']),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=prm['dropout']),
            nn.Linear(prm['fc1_neurons'], prm['fc2_neurons']),
            nn.LeakyReLU(inplace=True),
            nn.Linear(prm['fc2_neurons'], out_shape[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Chromosome used to generate this model:
# {'conv1_filters': 32, 'conv1_kernel': 11, 'conv1_stride': 4, 'conv2_filters': 128, 'conv2_kernel': 3, 'conv3_filters': 384, 'conv4_filters': 384, 'conv5_filters': 256, 'fc1_neurons': 3072, 'fc2_neurons': 2048, 'lr': 0.01, 'momentum': 0.95, 'dropout': 0.4, 'include_conv1': 0, 'include_conv2': 1, 'include_conv3': 1, 'include_conv4': 1, 'include_conv5': 1, 'pooling_type1': 'AdaptiveMaxPool2d', 'pooling_type2': 'MaxPool2d', 'pooling_type3': 'MaxPool2d', 'activation_type': 'LeakyReLU', 'use_batchnorm': 1}