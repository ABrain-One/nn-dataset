import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Net class with helper block: ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        out = self.layer(x)
        out += x
        return torch.relu(out)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.in_channels = in_shape[1]
        self.in_height = in_shape[2]
        self.in_width = in_shape[3]
        self.num_classes = out_shape[0]
        self.device = device

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(64 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.num_classes),
        )

        self._create_from_prm(prm)
    def _create_from_prm(self, prm):
        self.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'] if 'lr' in prm else 0.05, momentum=prm['momentum'] if 'momentum' in prm else 0.9)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

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

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return torch.relu(out)

class LegalClass:
    def __init__(self, db_handle):
        self.db_handle = db_handle

    def connect(self, host, port, user, password):
        return self.db_handle(host, port, user, password)

class DBConnector:
    def __init__(self, real_connect):
        self.real_connect = real_connect

    async def __aenter__(self):
        self.conn = await self.real_connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def execute(self, query):
        return await self.conn.execute(query)

    async def fetch(self, query):
        return await self.conn.fetch(query)

    async def close(self):
        await self.conn.close()

class DbClient:
    def __init__(self, db):
        self.db = db

    async def connect(self, uri):
        await self.db.set_uri(uri)

    async def close(self):
        await self.db.close()

    async def execute(self, query):
        return await self.db.execute(query)

    async def fetch(self, query):
        return await self.db.fetch(query)

class DbWrapper:
    def __init__(self, db):
        self.db = db

    async def set_uri(self, uri):
        await self.db.set_uri(uri)

    async def close(self):
        await self.db.close()

    async def execute(self, query):
        return await self.db.execute(query)

    async def fetch(self, query):
        return await self.db.fetch(query)

class AsyncDB:
    def __init__(self, loop):
        self.loop = loop
        self.pool = None

    async def connect(self, host, port, user, password, database):
        self.pool = await asyncpg.create_pool(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            loop=self.loop
        )

    async def close(self):
        await self.pool.close()

    async def set_uri(self, uri):
        self.pool = await asyncpg.create_pool(dsn=uri, loop=self.loop)

    async def execute(self, query, *args):
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query, *args):
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)