# Auto-generated single-file for BertEmbeddings_nopos
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn

# ---- original imports from contributing modules ----
from torch import nn

# ---- BertEmbeddings_nopos (target) ----
class BertEmbeddings_nopos(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id)
        # self.position_embeddings = nn.Embedding(
        #               config.max_position_embeddings, config.hidden_size)
        '''self.LayerNorm is not snake-cased to stick with
        TensorFlow model variable name and be able to load'''
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous
        # in memory and exported when serialized
        # self.register_buffer("position_ids",
        #       torch.arange(config.max_position_embeddings).expand((1, -1)))
        # self.position_embedding_type = \
        #           getattr(config, "position_embedding_type", "absolute")

        self.config = config

    def forward(self,
                input_ids=None,
                position_ids=None,
                inputs_embeds=None,
                past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]  # noqa: F841

        # if position_ids is None:
        #   position_ids = self.position_ids[:, \
        #       past_key_values_length : seq_length + \
        #       past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        # if self.position_embedding_type == "absolute":
        #     position_embeddings = self.position_embeddings(position_ids)
        #     # print('add position_embeddings!!!!')
        #     embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        layers += [
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        layers += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]

        class SimpleConfig:
            def __init__(self):
                self.vocab_size = 1000
                self.hidden_size = 128
                self.pad_token_id = 0
                self.layer_norm_eps = 1e-12
                self.hidden_dropout_prob = 0.1
        
        self.bert_embeddings_nopos = BertEmbeddings_nopos(SimpleConfig())
        
        layers += [
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]

        self._last_channels = 128
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        
        # Apply BertEmbeddings_nopos as an attention-like mechanism
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).transpose(1, 2)  # (B, H*W, C)
        
        # Create token IDs for spatial locations
        token_ids = torch.arange(H*W, device=x.device).unsqueeze(0).expand(B, -1) % 1000
        
        # Apply BertEmbeddings_nopos to get enhanced features
        x_embedded = self.bert_embeddings_nopos(token_ids)  # (B, H*W, 128)
        
        # Reshape back to spatial format and apply attention
        x_embedded = x_embedded.transpose(1, 2).view(B, 128, H, W)  # (B, 128, H, W)
        
        # Use the embedded features as attention weights
        attention_weights = torch.sigmoid(x_embedded)
        x_attended = x * attention_weights
        
        # Global average pooling and classification
        x = self.avgpool(x_attended)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)

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
