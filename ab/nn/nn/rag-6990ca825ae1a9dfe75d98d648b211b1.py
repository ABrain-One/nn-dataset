# Auto-generated single-file for BertEncoder
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from transformers import PretrainedConfig

# ---- src.transformers.models.bert.configuration_bert.BertConfig ----
class BertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BertModel`] or a [`TFBertModel`]. It is used to
    instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BERT
    [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://huggingface.co/papers/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://huggingface.co/papers/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from transformers import BertConfig, BertModel

    >>> # Initializing a BERT google-bert/bert-base-uncased style configuration
    >>> configuration = BertConfig()

    >>> # Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration
    >>> model = BertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

# ---- mmdet.models.language_models.bert.HFBertModel ----
try:
    from transformers import AutoTokenizer, BertConfig
    from transformers import BertModel as HFBertModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

# ---- BertEncoder (target) ----
class BertEncoder(nn.Module):
    """BERT encoder for language embedding.

    Args:
        name (str): name of the pretrained BERT model from HuggingFace.
                Defaults to bert-base-uncased.
        add_pooling_layer (bool): whether to add a pooling layer.
        num_layers_of_embedded (int): number of layers of the embedded model.
                Defaults to 1.
        use_checkpoint (bool): whether to use gradient checkpointing.
                Defaults to False.
    """

    def __init__(self,
                 name: str,
                 add_pooling_layer: bool = False,
                 num_layers_of_embedded: int = 1,
                 use_checkpoint: bool = False):
        super().__init__()
        if BertConfig is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')
        config = BertConfig.from_pretrained(name)
        config.gradient_checkpointing = use_checkpoint
        # only encoder
        self.model = HFBertModel.from_pretrained(
            name, add_pooling_layer=add_pooling_layer, config=config)
        self.language_dim = config.hidden_size
        self.num_layers_of_embedded = num_layers_of_embedded

    def forward(self, x) -> dict:
        mask = x['attention_mask']

        outputs = self.model(
            input_ids=x['input_ids'],
            attention_mask=mask,
            position_ids=x['position_ids'],
            token_type_ids=x['token_type_ids'],
            output_hidden_states=True,
        )

        # outputs has 13 layers, 1 input layer and 12 hidden layers
        encoded_layers = outputs.hidden_states[1:]
        features = torch.stack(encoded_layers[-self.num_layers_of_embedded:],
                               1).mean(1)
        # language embedding has shape [len(phrase), seq_len, language_dim]
        features = features / self.num_layers_of_embedded
        if mask.dim() == 2:
            embedded = features * mask.unsqueeze(-1).float()
        else:
            embedded = features

        results = {
            'embedded': embedded,
            'masks': mask,
            'hidden': encoded_layers[-1]
        }
        return results


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
        self.classifier = nn.Linear(64, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        self.bert_encoder = BertEncoder('bert-base-uncased', add_pooling_layer=False, num_layers_of_embedded=1, use_checkpoint=True)
        self.channel_proj = nn.Conv2d(768, 64, 1)

        self._last_channels = 64
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        
        x_small = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        B, C, H_small, W_small = x_small.shape
        
        input_ids = torch.arange(H_small*W_small, device=x.device).unsqueeze(0).expand(B, -1) % 1000
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(H_small*W_small, device=x.device).unsqueeze(0).expand(B, -1) % 512
        token_type_ids = torch.zeros_like(input_ids)
        
        bert_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids
        }
        
        bert_output = self.bert_encoder(bert_input)
        x_embedded = bert_output['embedded']
        
        x_embedded = x_embedded.transpose(1, 2).view(B, 768, H_small, W_small)
        
        # Project to match input channels first
        x_embedded = self.channel_proj(x_embedded)
        
        # Then upsample
        x_embedded = torch.nn.functional.interpolate(x_embedded, size=(H, W), mode='bilinear', align_corners=False)
        
        attention_weights = torch.sigmoid(x_embedded)
        x_attended = x * attention_weights
        
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
