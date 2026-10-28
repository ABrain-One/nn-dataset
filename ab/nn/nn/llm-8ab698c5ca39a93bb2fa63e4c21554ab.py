import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        c_in = 3
        n_classes = out_shape[0] if out_shape else 10
        start_chan = 64
        self.entry = nn.Sequential(nn.Conv2d(c_in, start_chan, kernel_size=3, padding=1), nn.BatchNorm2d(start_chan), nn.ReLU(inplace=True))
        blocks = []
        pools = []
        trans_layers = []
        cur_chan = start_chan
        total_blocks = 3
        for i in range(total_blocks):
            blocks.append(nn.Sequential(nn.Conv2d(cur_chan, cur_chan, kernel_size=3, padding=1), nn.BatchNorm2d(cur_chan), nn.ReLU(inplace=True), nn.Conv2d(cur_chan, cur_chan * 2, kernel_size=3, padding=1), nn.BatchNorm2d(cur_chan * 2), nn.ReLU(inplace=True)))
            pools.append(nn.MaxPool2d(2))
            cur_chan *= 2
        self.blocks = nn.ModuleList(blocks)
        self.pools = nn.ModuleList(pools)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(cur_chan, n_classes)
        self.to(device)

    def forward(self, x):
        x = self.entry(x)
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            x = pool(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm['momentum'], 0.999), weight_decay=0.0001)
        self.max_batches = prm.get('max_batches', None)
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
            if self.max_batches is not None and i >= self.max_batches:
                break
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()