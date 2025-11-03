import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils

class Net(nn.Module):
    def __init__(self, inputs, input_channels, output_channels, num_classes):
        super(Net, self).__init__()
        self.conv1 = BBBConv2d(inputs, 64, ...)
        ...
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, ...),
            ...
        )
        self.fc = nn.Linear(output_channels, num_classes)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999))

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images, captions = images.to(self.device), captions.to(self.device)
            logits = None
            if hasattr(self, 'forward'):
                out = self.forward(images, captions)
                logits = out[0] if isinstance(out, tuple) else out
            if logits is None:
                continue
            tgt = (captions[:,0,:] if captions.ndim==3 else captions)[:,1:]
            loss = self.criteria[0](logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def forward(self, x, y):
        # Teacher forcing implementation
        ...
        # Shape asserts
        ...
        return ...

def supported_hyperparameters():
    return {'lr','momentum'}