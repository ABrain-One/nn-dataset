# Auto-generated single-file for BertOutput
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# object
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
from torch import Tensor

# ---- mmdet.models.utils.vlfuse_helper.HFBertOutput ----
try:
    from transformers import BertConfig, BertPreTrainedModel
    from transformers.modeling_utils import apply_chunking_to_forward
    from transformers.models.bert.modeling_bert import \
        BertAttention as HFBertAttention
    from transformers.models.bert.modeling_bert import \
        BertIntermediate as HFBertIntermediate
    from transformers.models.bert.modeling_bert import \
        BertOutput as HFBertOutput
except ImportError:
    BertConfig = None
    BertPreTrainedModel = object
    apply_chunking_to_forward = None
    HFBertAttention = object
    HFBertIntermediate = object
    
    class HFBertOutput(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
            self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

# ---- mmdet.models.utils.vlfuse_helper.MAX_CLAMP_VALUE ----
MAX_CLAMP_VALUE = 50000

# ---- mmdet.models.utils.vlfuse_helper.clamp_values ----
def clamp_values(vector: Tensor) -> Tensor:
    """Clamp the values of a vector to the range [-MAX_CLAMP_VALUE,
    MAX_CLAMP_VALUE].

    Args:
        vector (Tensor): Tensor of shape (N, C, H, W).

    Returns:
        Tensor: A Tensor of shape (N, C, H, W) with clamped values.
    """
    vector = torch.clamp(vector, min=-MAX_CLAMP_VALUE, max=MAX_CLAMP_VALUE)
    return vector

# ---- BertOutput (target) ----
class BertOutput(HFBertOutput):
    """Modified from transformers.models.bert.modeling_bert.BertOutput.

    Compared to the BertOutput of Huggingface, only add the clamp.
    """

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = clamp_values(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = clamp_values(hidden_states)
        return hidden_states


def supported_hyperparameters():
    return {'lr','momentum'}


class Net(torch.nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]

        layers += [
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]

        class SimpleConfig:
            def __init__(self):
                self.hidden_size = 32
                self.hidden_dropout_prob = 0.1
                self.layer_norm_eps = 1e-12
        
        self.bert_output = BertOutput(SimpleConfig())
        
        layers += [
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]

        self._last_channels = 32
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        
        x_flat = x.view(B, C, H*W).transpose(1, 2)
        
        x_embedded = self.bert_output(x_flat, x_flat)
        
        x_embedded = x_embedded.transpose(1, 2).view(B, 32, H, W)
        
        attention_weights = torch.sigmoid(x_embedded)
        x_attended = x * attention_weights
        
        x = self.avgpool(x_attended)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
