from abc import ABCMeta, abstractmethod
from typing import Collection, Dict, List, NamedTuple, Optional, Union
import numpy as np
import torch
from torch import nn
import itertools
import sys
import warnings
from collections.abc import Sequence
from typing import List, Optional, Set, Tuple, Union
import functools
import inspect
import threading
from typing import ContextManager

class AdapterCompositionBlock(Sequence):
    def __init__(self, *children):
        self.children = [parse_composition(b, None) for b in children]

    def __getitem__(self, key):
        return self.children[key]

    def __len__(self):
        return len(self.children)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, type(self)):
            return all([c1 == c2 for c1, c2 in zip(self.children, o.children)])
        else:
            return False

    def __repr__(self):
        child_repr = ", ".join(map(str, self.children))
        return f"{self.__class__.__name__}[{child_repr}]"

    def first(self):
        if not isinstance(self.children[0], AdapterCompositionBlock):
            return self.children[0]
        else:
            return self.children[0].first()

    def last(self):
        if not isinstance(self.children[-1], AdapterCompositionBlock):
            return self.children[-1]
        else:
            return self.children[-1].last()

    @property
    def parallel_channels(self):
        return max([(b.parallel_channels if isinstance(b, AdapterCompositionBlock) else 1) for b in self.children])

    def flatten(self) -> Set[str]:
        return set(itertools.chain(*[[b] if isinstance(b, str) else b.flatten() for b in self.children]))

    def _get_save_kwargs(self):
        return None

    def to_dict(self):
        save_dict = {
            "type": self.__class__.__name__,
            "children": [
                (c.to_dict() if isinstance(c, AdapterCompositionBlock) else {"type": "single", "children": [c]})
                for c in self.children
            ],
        }
        if kwargs := self._get_save_kwargs():
            save_dict["kwargs"] = kwargs
        return save_dict

    @classmethod
    def from_dict(cls, data):
        children = []
        for child in data["children"]:
            if child["type"] == "single":
                children.append(child["children"][0])
            else:
                children.append(cls.from_dict(child))
        return getattr(sys.modules[__name__], data["type"])(*children, **data.get("kwargs", {}))


class Parallel(AdapterCompositionBlock):
    def __init__(self, *parallel_adapters: List[str]):
        """
        Can be used to perform inference for multiple tasks (i.e., adapters) in parallel (for the same input).

        See AdapterDrop https://arxiv.org/abs/2010.11918
        """
        super().__init__(*parallel_adapters)

    @property
    def parallel_channels(self):
        return len(self.children)


class Stack(AdapterCompositionBlock):
    def __init__(self, *stack_layers: List[Union[AdapterCompositionBlock, str]]):
        super().__init__(*stack_layers)


class Fuse(AdapterCompositionBlock):
    def __init__(
        self,
        *fuse_stacks: List[Union[AdapterCompositionBlock, str]],
        name: Optional[str] = None,
    ):
        super().__init__(*fuse_stacks)
        self._name = name

    # TODO-V2 pull this up to all block classes?
    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return ",".join([c if isinstance(c, str) else c.last() for c in self.children])


class Split(AdapterCompositionBlock):
    def __init__(
        self,
        *split_adapters: List[Union[AdapterCompositionBlock, str]],
        splits: Union[List[int], int],
    ):
        super().__init__(*split_adapters)
        self.splits = splits if isinstance(splits, list) else [splits] * len(split_adapters)

    def _get_save_kwargs(self):
        return {"splits": self.splits}


class BatchSplit(AdapterCompositionBlock):
    def __init__(
        self,
        *split_adapters: List[Union[AdapterCompositionBlock, str]],
        batch_sizes: Union[List[int], int],
    ):
        super().__init__(*split_adapters)
        self.batch_sizes = batch_sizes if isinstance(batch_sizes, list) else [batch_sizes] * len(split_adapters)

    def _get_save_kwargs(self):
        return {"batch_sizes": self.batch_sizes}


class MultiTask(AdapterCompositionBlock):
    def __init__(self, *children):
        super().__init__(*children)


class Average(AdapterCompositionBlock):
    def __init__(
        self,
        *average_adapters: List[Union[AdapterCompositionBlock, str]],
        weights: Optional[List[float]] = None,
        normalize_weights: bool = True,
    ):
        super().__init__(*average_adapters)
        if weights is not None:
            # normalize weights
            if normalize_weights:
                sum_weights = sum(weights) if weights else 1
                self.weights = [w / sum_weights for w in weights]
            else:
                self.weights = weights
        else:
            self.weights = [1 / len(average_adapters)] * len(average_adapters)

    def _get_save_kwargs(self):
        return {"weights": self.weights}


# Mapping each composition block type to the allowed nested types
ALLOWED_NESTINGS = {
    Stack: [str, Fuse, Split, Parallel, BatchSplit, Average, MultiTask],
    Fuse: [str, Stack],
    Split: [str, Split, Stack, BatchSplit, Average],
    Parallel: [str, Stack, BatchSplit, Average],
    MultiTask: [str, Stack, Average, Fuse],
    BatchSplit: [str, Stack, Split, BatchSplit, Average],
    Average: [str, Stack, Split, BatchSplit],
}

# Some composition blocks might not be supported by all models.
# Add a whitelist of models for those here.
SUPPORTED_MODELS = {
    Parallel: [
        "albert",
        "bert",
        "roberta",
        "distilbert",
        "deberta-v2",
        "deberta",
        "bart",
        "mbart",
        "mt5",
        "plbart",
        "gpt2",
        "gptj",
        "t5",
        "vit",
        "xlm-roberta",
        "bert-generation",
        "llama",
        "mistral",
        "electra",
        "whisper",
        "xmod",
    ],
}


def validate_composition(adapter_composition: AdapterCompositionBlock, level=0, model_type=None):
    if level > 1 and not (isinstance(adapter_composition, Stack) or isinstance(adapter_composition, str)):
        raise ValueError(f"Adapter setup is too deep. Cannot have {adapter_composition} at level {level}.")
    if isinstance(adapter_composition, AdapterCompositionBlock):
        block_type = type(adapter_composition)
        if model_type and block_type in SUPPORTED_MODELS:
            if model_type not in SUPPORTED_MODELS[block_type]:
                raise ValueError(
                    f"Models of type {model_type} don't support adapter composition using {block_type.__name__}."
                )
        for child in adapter_composition:
            if not type(child) in ALLOWED_NESTINGS[type(adapter_composition)]:
                raise ValueError(f"Adapter setup is invalid. Cannot nest {child} in {adapter_composition}")
            # recursively validate children
            validate_composition(child, level=level + 1)


def parse_composition(adapter_composition, level=0, model_type=None) -> AdapterCompositionBlock:
    """
    Parses and validates a setup of adapters.

    Args:
        adapter_composition: The adapter setup to be parsed.
        level (int, optional): If set to none, disables validation. Defaults to 0.
    """
    if not adapter_composition:
        return None
    elif isinstance(adapter_composition, AdapterCompositionBlock):
        if level is not None:
            validate_composition(adapter_composition, level=level, model_type=model_type)
        return adapter_composition
    elif isinstance(adapter_composition, str):
        if level == 0:
            return Stack(adapter_composition)
        else:
            return adapter_composition
    elif isinstance(adapter_composition, Sequence):
        # Functionality of adapter-transformers v1.x
        warnings.warn(
            "Passing list objects for adapter activation is deprecated. Please use Stack or Fuse explicitly.",
            category=FutureWarning,
        )
        # for backwards compatibility
        if level == 1:
            block_class = Fuse
        else:
            block_class = Stack
        level = level + 1 if level is not None else None
        return block_class(*[parse_composition(b, level) for b in adapter_composition])
    else:
        raise TypeError(adapter_composition)


def parse_heads_from_composition(adapter_composition, reference_heads: list = None):
    """
    Parses a potential head configuration from a setup of adapters.

    Args:
        adapter_composition: The adapter setup to be parsed.
        reference_heads: The list of available to validate the retrieved head configuration against.
    """
    final_block = adapter_composition
    if isinstance(final_block, Stack):
        final_block = final_block.children[-1]

    if isinstance(final_block, str) and (reference_heads is None or final_block in reference_heads):
        return final_block
    elif isinstance(final_block, Parallel):
        return [a if isinstance(a, str) else a.last() for a in final_block.children]
    elif isinstance(final_block, BatchSplit):
        # Convert BatchSplit of adapters to a BatchSplit of heads.
        blocks = [(block.last() if isinstance(block, AdapterCompositionBlock) else block) for block in final_block]
        head_setup = BatchSplit(*blocks, batch_sizes=final_block.batch_sizes)
        if reference_heads is None or all(head in reference_heads for head in head_setup):
            return head_setup
        else:
            raise ValueError(
                "Missing at least one head for the given BatchSplit setup. Expected heads: {}".format(blocks)
            )
    else:
        return None


def adjust_tensors_for_parallel(hidden_states, *tensors):
    """
    Replicates a given list of tensors based on the shape of the reference tensor (first argument).
    """
    outputs = []
    for tensor in tensors:
        if tensor is not None and hidden_states.shape[0] > tensor.shape[0]:
            repeats = [1] * len(tensor.shape)
            repeats[0] = hidden_states.shape[0] // tensor.shape[0]
            new_tensor = tensor.repeat(*repeats)
            outputs.append(new_tensor)
        else:
            outputs.append(tensor)
    return tuple(outputs)


def adjust_tensors_for_parallel_(hidden_states, *tensors):
    """
    In-place version of adjust_tensors_for_parallel().
    """
    for tensor in tensors:
        if tensor is not None and hidden_states.shape[0] > tensor.shape[0]:
            repeats = [1] * len(tensor.shape)
            repeats[0] = hidden_states.shape[0] // tensor.shape[0]
            new_tensor = tensor.repeat(*repeats)
            tensor.set_(new_tensor)


def match_attn_matrices_for_parallel(query, key, value) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Matches the shapes of query, key and value matrices for parallel composition.
    """
    max_bsz = max(query.shape[0], key.shape[0], value.shape[0])

    query = query.repeat(max_bsz // query.shape[0], *([1] * len(query.shape[1:])))
    key = key.repeat(max_bsz // key.shape[0], *([1] * len(key.shape[1:])))
    value = value.repeat(max_bsz // value.shape[0], *([1] * len(value.shape[1:])))

    return query, key, value

class AdapterSetup(ContextManager):
    """
    Represents an adapter setup of a model including active adapters and active heads. This class is intended to be
    used as a context manager using the ``with`` statement. The setup defined by the ``AdapterSetup`` context will
    override static adapter setups defined in a model (i.e. setups specified via ``active_adapters``).

    Example::

        with AdapterSetup(Stack("a", "b")):
            # will use the adapter stack "a" and "b" outputs = model(**inputs)

    Note that the context manager is thread-local, i.e. it can be used with different setups in a multi-threaded
    environment.
    """

    # thread-local storage that holds a stack of active contexts
    storage = threading.local()

    def __init__(self, adapter_setup, head_setup=None, ignore_empty: bool = False):
        self.adapter_setup = parse_composition(adapter_setup)
        if head_setup:
            self.head_setup = head_setup
        else:
            self.head_setup = parse_heads_from_composition(self.adapter_setup)
        self._empty = ignore_empty and self.adapter_setup is None and self.head_setup is None

    def __enter__(self):
        if not self._empty:
            AdapterSetup.get_contexts().append(self)
        return self

    def __exit__(self, type, value, traceback):
        if not self._empty:
            AdapterSetup.get_contexts().pop()

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls.storage, "contexts"):
            cls.storage.contexts = []
        return cls.storage.contexts

    @classmethod
    def get_context(cls):
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            return None

    @classmethod
    def get_context_adapter_setup(cls):
        context = cls.get_context()
        if context:
            return context.adapter_setup
        return None

    @classmethod
    def get_context_head_setup(cls):
        context = cls.get_context()
        if context:
            return context.head_setup
        return None


class ForwardContext(ContextManager):
    """
    Holds context information during a forward pass through a model. This class should be used via the
    ``ForwardContext.wrap()`` method.

    Note that the context is thread-local.
    """

    # thread-local storage that holds a stack of active contexts
    storage = threading.local()

    context_args = {
        "output_adapter_gating_scores",
        "output_adapter_fusion_attentions",
        "adapter_input_parallelized",
        "task_ids",
    }
    context_attributes = {
        "adapter_gating_scores",
        "adapter_fusion_attentions",
    }
    # Additional used attributes not exposed to the user
    # - prompt_tokens_length: length of the prompt tokens

    def __init__(self, model, *args, **kwargs):
        # If the model has a method ``forward_context()``, use it to create the context.
        for arg_name in self.context_args:
            setattr(self, arg_name, kwargs.pop(arg_name, None))
        if hasattr(model, "forward_context"):
            model.forward_context(self, *args, **kwargs)

    def __enter__(self):
        ForwardContext.get_contexts().append(self)
        return self

    def __exit__(self, type, value, traceback):
        ForwardContext.get_contexts().pop()

    def _call_forward(self, model, f, *args, **kwargs):
        """
        Calls the forward function of the model with the given arguments and keyword arguments.
        """
        kwargs = {k: v for k, v in kwargs.items() if k not in self.context_args}
        results = f(model, *args, **kwargs)

        # append output attributes
        if isinstance(results, tuple):
            for attr in self.context_attributes:
                if getattr(self, "output_" + attr, False):
                    results = results + (dict(getattr(self, attr)),)
        else:
            for attr in self.context_attributes:
                if getattr(self, "output_" + attr, False):
                    results[attr] = dict(getattr(self, attr))

        return results

    @classmethod
    def add_context_args_in_signature(cls, f):
        old_signature = inspect.signature(f)
        params = list(old_signature.parameters.values())
        # search if a VAR_POSITIONAL or VAR_KEYWORD is present
        # if yes insert step parameter before it, else insert it in last position
        param_types = [param.kind for param in params]
        i = min(
            [
                (param_types.index(param_type) if param_type in param_types else float("inf"))
                for param_type in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
            + [len(params)]
        )
        for name in cls.context_args:
            new_param = inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)
            if new_param not in params:
                params.insert(i, new_param)
            # we can now build the signature for the wrapper function
        new_signature = old_signature.replace(parameters=params)
        return new_signature

    @classmethod
    def wrap_base(cls, f):
        """
        Decorator method that wraps a ``forward()`` function of a base model class.
        Unlike ``wrap()``, this method does not create a new context if the is an existing one.
        """

        @functools.wraps(f)
        def wrapper_func(self, *args, **kwargs):
            if self.adapters_config is not None and ForwardContext.get_context() is None:
                with cls(self, *args, **kwargs) as ctx:
                    results = ctx._call_forward(self, f, *args, **kwargs)
                return results
            else:
                return f(self, *args, **kwargs)

        return wrapper_func

    @classmethod
    def wrap(cls, f):
        """
        Decorator method that wraps a ``forward()`` function of a model class.
        """

        @functools.wraps(f)
        def wrapper_func(self, *args, **kwargs):
            if self.adapters_config is not None:
                with cls(self, *args, **kwargs) as ctx:
                    results = ctx._call_forward(self, f, *args, **kwargs)
                return results
            else:
                return f(self, *args, **kwargs)

        return wrapper_func

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls.storage, "contexts"):
            cls.storage.contexts = []
        return cls.storage.contexts

    @classmethod
    def get_context(cls):
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            return None

import torch, torch.nn as nn

class AdapterLayerBase(metaclass=ABCMeta):
    """
    Base class for all adaptation methods that require per-layer modules.

    Make sure the 'adapter_modules_name' attribute is overriden in derived classes.
    """

    adapter_modules_name = ""

    @property
    def adapter_modules(self) -> Collection:
        return getattr(self, self.adapter_modules_name)

    @property
    def layer_idx(self):
        return getattr(self, "_layer_idx", -1)

    @layer_idx.setter
    def layer_idx(self, layer_idx):
        idx = getattr(self, "_layer_idx", layer_idx)
        assert idx == layer_idx
        setattr(self, "_layer_idx", idx)

    def get_active_setup(self):
        if hasattr(self, "adapters_config"):
            # First check current context before falling back to defined setup
            context = AdapterSetup.get_context()
            if context is not None:
                adapter_setup = context.adapter_setup
            else:
                adapter_setup = self.adapters_config.active_setup
        else:
            adapter_setup = None
        skip_adapters = adapter_setup is None or (
            self.adapters_config.skip_layers is not None and self.layer_idx in self.adapters_config.skip_layers
        )
        if not skip_adapters and (len(set(self.adapter_modules.keys()) & adapter_setup.flatten()) > 0):
            return adapter_setup
        else:
            return None

    def _store_gating_score(self, adapter_name, gating_score):
        context = ForwardContext.get_context()
        if context.output_adapter_gating_scores:
            gating_cache = context.adapter_gating_scores
            if self.layer_idx not in gating_cache[adapter_name]:
                gating_cache[adapter_name][self.layer_idx] = {}
            gating_score = gating_score.detach().squeeze().cpu().numpy()
            if len(gating_score.shape) == 0:
                gating_score = np.expand_dims(gating_score, axis=0)
            cache_score = gating_cache[adapter_name][self.layer_idx].get(self.location_key, None)
            if cache_score is not None:
                gating_cache[adapter_name][self.layer_idx][self.location_key] = np.column_stack(
                    (cache_score, gating_score)
                )
            else:
                gating_cache[adapter_name][self.layer_idx][self.location_key] = gating_score

    def _store_fusion_attentions(self, fusion_name, attentions):
        context = ForwardContext.get_context()
        if context.output_adapter_fusion_attentions:
            attention_cache = context.adapter_fusion_attentions
            if self.layer_idx not in attention_cache[fusion_name]:
                attention_cache[fusion_name][self.layer_idx] = {}
            attention_cache[fusion_name][self.layer_idx][self.location_key] = attentions

    @abstractmethod
    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        """Adds a new adapter module to the layer.

        Args:
            adapter_name (str): The name of the new adapter to add.
            layer_idx (int):
                The index of the adapters layer (this should be set once by the first added adapter and the kept fix).

        Returns:
            bool: True if the adapter was added, False otherwise.
        """
        raise NotImplementedError()

    def average_adapter(
        self,
        adapter_name: str,
        input_adapters: Dict[str, float],
        combine_strategy,
        **kwargs,
    ) -> bool:
        """Averages a set of adapter modules into a new adapter module.

        Args:
            adapter_name (str): The name of the new (averaged) adapter module to add.
            input_adapters (Dict[str, float]): Dictionary of adapter names and their corresponding weights.
            combine_strategy (str): The strategy to combine the adapters. Available strategies depend on the used adapter method, see: https://docs.adapterhub.ml/adapter_composition.html#merging-adapters
            **kwargs: Additional arguments that are specific to the combine_strategy. E.g. svd_rank for LoRA.

        Returns:
            bool: True if the adapter was added, False otherwise.
        """
        # add new adapter
        if self.add_adapter(adapter_name, self.layer_idx):
            if combine_strategy != "linear":
                # You get the adapter type from the input adapters
                raise ValueError(f"Combine strategy {combine_strategy} not supported for the chosen adapter methods.")

            # average weights linearly
            avg_state_dict = {}
            for name, weight in input_adapters.items():
                if name in self.adapter_modules:
                    module = self.adapter_modules[name]
                    for k, v in module.state_dict().items():
                        if k in avg_state_dict:
                            avg_state_dict[k] += weight * v
                        else:
                            avg_state_dict[k] = weight * v
                else:
                    self.delete_adapter(adapter_name)  # clean up before raising error
                    raise ValueError("Adapter {} not found.".format(name))

            # load averaged weights
            self.adapter_modules[adapter_name].load_state_dict(avg_state_dict)

            return True

        return False

    def delete_adapter(self, adapter_name: str):
        """Deletes an adapter module from the layer.

        Args:
            adapter_name (str): The name of the adapter to delete.
        """
        if adapter_name in self.adapter_modules:
            del self.adapter_modules[adapter_name]

    def share_parameters(
        self,
        name: str,
        adapter_names: List,
        reference_adapter_name: Optional[str],
    ):
        pass  # default implementation does nothing as multi task is not applicable to all methods

    def unshare_parameters(self, name: str):
        pass  # default implementation does nothing as multi task is not applicable to all methods

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # default implementation does nothing as fusion is not applicable to all methods

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # default implementation does nothing as fusion is not applicable to all methods

    def enable_adapters(
        self,
        adapter_setup: AdapterCompositionBlock,
        unfreeze_adapters: bool,
        unfreeze_fusion: bool,
    ):
        """Enables/ disables a set of adapter modules within the layer.

        Args:
            adapter_setup (AdapterCompositionBlock): The adapter setup to enable/ disable.
            unfreeze_adapters (bool): Whether to unfreeze the adapters.
        """
        if unfreeze_adapters:
            for name in adapter_setup.flatten():
                if name in self.adapter_modules:
                    for param in self.adapter_modules[name].parameters():
                        param.requires_grad = True

    def freeze_adapter(self, adapter_name: str, freeze: bool = True):
        """Freezes/ unfreezes an adapter module.

        Args:
            adapter_name (str): The name of the adapter to freeze/ unfreeze.
            freeze (bool, optional): Whether to freeze the adapter. Defaults to True.
        """
        if adapter_name in self.adapter_modules:
            self.adapter_modules[adapter_name].train(not freeze)
            for param in self.adapter_modules[adapter_name].parameters():
                param.requires_grad = not freeze

    def get_adapter(self, adapter_name: str) -> nn.Module:
        """Returns the adapter module with the given name.

        Args:
            adapter_name (str): The name of the adapter module.
        """
        if adapter_name in self.adapter_modules:
            return self.adapter_modules[adapter_name]
        else:
            return None

    def pre_save_adapters(self):
        """Called before saving the adapters to disk."""
        pass


import torch.nn as nn
import torch.nn.functional as F

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
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        self._last_channels = 32
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()