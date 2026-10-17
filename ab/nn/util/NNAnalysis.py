import json
from pathlib import Path
from typing import Optional, Any, Dict

import torch
import torch.nn as nn

from ab.nn.api import data
from ab.nn.util.Const import stat_nn_dir
from ab.nn.util.Loader import load_dataset
from ab.nn.util.Util import get_in_shape, torch_device, first_tensor

try:
    from ab.nn.util.NNAnalysisSimilarity import to_minhash
    _HAS_DATASKETCH = True
except Exception:
    to_minhash = None
    _HAS_DATASKETCH = False

def get_max_depth(module, depth=0):
    '''Calculate maximum depth of the model.'''
    children = list(module.children())
    if not children:
        return depth
    return max(get_max_depth(child, depth + 1) for child in children)


# Ops that always carry learnable weights when they appear in a graph.
_DEPTH_OPS = ('ConvolutionBackward', 'ConvolutionTransposeBackward',
              'CudnnConvolutionBackward', 'MkldnnConvolutionBackward',
              'SlowConvTranspose', 'ThnnConv', 'ConvDepthwise',
              'AddmmBackward', 'LinearBackward', 'AddbmmBackward',
              'CudnnRnn', 'MkldnnRnnLayer', 'ThnnFused', 'RnnTanh', 'RnnRelu',
              'Lstm', 'Gru',
              'EmbeddingBackward', 'EmbeddingBag')

# Ops that carry weights only sometimes. `q @ k.transpose(-2,-1)` and
# `weight @ x` are both MmBackward, but only the second is a layer. These are
# counted only when a model parameter feeds into them.
_MAYBE_DEPTH_OPS = ('MmBackward', 'BmmBackward', 'MatmulBackward',
                    'BaddbmmBackward', 'EinsumBackward', 'TensordotBackward')

# Shape-only nodes a weight may pass through before reaching a matmul
# (nn.Linear routes its weight through TBackward, for instance).
_TRANSPARENT = ('TBackward', 'Transpose', 'View', 'Reshape', 'Alias',
                'Permute', 'Expand', 'Clone', 'ToCopy', 'Squeeze',
                'Unsqueeze', 'Slice', 'Narrow', 'AsStrided', 'Contiguous',
                'Select', 'Chunk', 'Split', 'Repeat', 'Flatten', 'Unflatten',
                'Copy', 'Cat', 'Stack')


def _param_feeds(node, param_ids, max_hops=8):
    '''True if one of the model's own parameters is an input to `node`,
    following only shape-changing nodes on the way.'''
    frontier = [(node, 0)]
    seen = set()
    while frontier:
        nd, hop = frontier.pop()
        if nd is None or hop > max_hops or id(nd) in seen:
            continue
        seen.add(id(nd))
        for nxt, _ in getattr(nd, 'next_functions', ()):
            if nxt is None:
                continue
            nm = type(nxt).__name__
            if nm == 'AccumulateGrad':
                v = getattr(nxt, 'variable', None)
                if v is not None and id(v) in param_ids:
                    return True
            elif any(k in nm for k in _TRANSPARENT):
                frontier.append((nxt, hop + 1))
    return False


def _output_tensors(out, acc=None):
    '''Every floating-point tensor inside an arbitrary model output.'''
    if acc is None:
        acc = []
    if torch.is_tensor(out):
        if out.is_floating_point():
            acc.append(out)
    elif isinstance(out, dict):
        for o in out.values():
            _output_tensors(o, acc)
    elif isinstance(out, (list, tuple)):
        for o in out:
            _output_tensors(o, acc)
    return acc


def get_nn_depth(model, input_tensor) -> Optional[int]:
    '''Real network depth: the longest chain of weight-bearing operations from
    input to output, measured on the autograd graph.

    This is the quantity the name "ResNet-50" refers to. It differs from
    get_max_depth, which is the nesting depth of the nn.Module tree (ResNet-50
    nests only ~4 levels), and from total_layers, which counts modules whether
    or not they lie on the same path.

    Verified exact on AlexNet 8, VGG-16 16, ResNet-18 18, ResNet-50 50,
    DenseNet-121 121, ViT-B/16 50.

    Returns None when no depth can be measured, so the caller can omit the
    field rather than store a sentinel. That happens when the forward pass
    fails at this input size, when forward() needs more than one tensor, or
    when the output is not differentiable (e.g. the model returns argmax
    indices), and when no weight-bearing operation is found on any path at all.
    This function never raises: a failure here must not cost the other
    statistics.

    Caveat: a forward pass that wraps part of the network in torch.no_grad()
    hides those layers from the autograd graph, so the result is the longest
    *differentiable* path and will undercount. Freezing with
    requires_grad=False is handled correctly and does not undercount.
    '''
    was_training = model.training
    model.eval()
    try:
        param_ids = {id(p) for p in model.parameters()}
        x = input_tensor.detach().clone().float().requires_grad_(True)
        with torch.enable_grad():
            out = model(x)
        roots = [t.grad_fn for t in _output_tensors(out) if t.grad_fn is not None]
        if not roots:
            return None
        # iterative longest path over the autograd DAG; recursion overflows on
        # deep graphs such as DenseNet
        memo = {}
        for root in roots:
            stack = [(root, False)]
            while stack:
                node, done = stack.pop()
                if node is None:
                    continue
                if done:
                    best = 0
                    for nxt, _ in node.next_functions:
                        if nxt is not None:
                            best = max(best, memo.get(nxt, 0))
                    name = type(node).__name__
                    if any(k in name for k in _DEPTH_OPS):
                        weighted = True
                    elif any(k in name for k in _MAYBE_DEPTH_OPS):
                        weighted = _param_feeds(node, param_ids)
                    else:
                        weighted = False
                    memo[node] = best + (1 if weighted else 0)
                elif node not in memo:
                    stack.append((node, True))
                    for nxt, _ in node.next_functions:
                        if nxt is not None and nxt not in memo:
                            stack.append((nxt, False))
        depth = int(max(memo.get(r, 0) for r in roots))
        return depth if depth > 0 else None
    except Exception:
        return None
    finally:
        model.train(was_training)


def analyze_conv_layers(model):
    '''Analyze convolutional layers.'''
    conv_layers = [m for m in model.modules()
                   if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d))]

    if not conv_layers:
        return {}

    kernel_sizes = []
    strides = []
    padding_values = []

    for m in conv_layers:
        # Handle tuple or int kernel sizes
        k = m.kernel_size
        kernel_sizes.append(k[0] if isinstance(k, tuple) else k)

        s = m.stride
        strides.append(s[0] if isinstance(s, tuple) else s)

        p = m.padding
        padding_values.append(p[0] if isinstance(p, tuple) else p)

    return {
        'count': len(conv_layers),
        'kernel_sizes': kernel_sizes,
        'strides': strides,
        'padding_values': padding_values,
        'avg_kernel_size': sum(kernel_sizes) / len(kernel_sizes) if kernel_sizes else 0,
        'avg_stride': sum(strides) / len(strides) if strides else 0,
        'total_conv_params': sum(p.numel() for m in conv_layers for p in m.parameters())
    }


def analyze_linear_layers(model):
    '''Analyze linear/dense layers.'''
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

    if not linear_layers:
        return {}

    return {
        'count': len(linear_layers),
        'input_dims': [m.in_features for m in linear_layers],
        'output_dims': [m.out_features for m in linear_layers],
        'total_linear_params': sum(p.numel() for m in linear_layers
                                   for p in m.parameters()),
        'has_bias': [m.bias is not None for m in linear_layers]
    }


def has_residual_connections(model):
    '''Detect if model has residual/skip connections.'''
    # Check for common residual patterns
    for module in model.modules():
        # Check if module name suggests residual
        module_name = type(module).__name__.lower()
        if any(keyword in module_name for keyword in ['residual', 'resnet', 'skip', 'shortcut']):
            return True

        # Check for Identity layers (common in residual blocks)
        if isinstance(module, nn.Identity):
            return True

    return False


def estimate_flops(model, input_tensor):
    '''Rough FLOPs estimation for common layers.'''
    flops = 0

    def hook(module, input, output):
        nonlocal flops

        if isinstance(module, nn.Conv2d):
            # FLOPs for Conv2d: 2 * kernel_h * kernel_w * in_channels * out_channels * out_h * out_w
            batch_size, in_channels, in_h, in_w = input[0].shape
            out_channels, _, kernel_h, kernel_w = module.weight.shape
            out_h, out_w = output.shape[2:]
            flops += 2 * kernel_h * kernel_w * in_channels * out_channels * out_h * out_w

        elif isinstance(module, nn.Linear):
            # FLOPs for Linear: 2 * in_features * out_features
            in_features = module.in_features
            out_features = module.out_features
            batch_size = input[0].shape[0]
            flops += 2 * in_features * out_features * batch_size

        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm FLOPs: 2 * num_features * H * W (mean and variance computation)
            batch_size, num_features, h, w = output.shape
            flops += 2 * num_features * h * w * batch_size

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    return flops


def analyze_compute_characteristics(model: nn.Module, input_tensor) -> dict:
    '''Analyze computational requirements.'''

    # FLOPs estimation
    flops = estimate_flops(model, input_tensor)

    # Memory footprint
    model_size_mb = sum(p.numel() * p.element_size()
                        for p in model.parameters()) / (1024 ** 2)

    # Calculate buffer memory (e.g., BatchNorm running stats)
    buffer_size_mb = sum(b.numel() * b.element_size()
                         for b in model.buffers()) / (1024 ** 2)

    return {
        'flops': flops,
        'model_size_mb': model_size_mb,
        'buffer_size_mb': buffer_size_mb,
        'total_memory_mb': model_size_mb + buffer_size_mb,
    }


def detect_architecture_patterns(model: nn.Module, nn_code: str) -> dict:
    '''Detect high-level architecture patterns.'''
    code_lower = nn_code.lower()

    return {
        'is_resnet_like': 'residual' in code_lower or 'resnet' in code_lower,
        'is_vgg_like': 'vgg' in code_lower,
        'is_inception_like': 'inception' in code_lower,
        'is_densenet_like': 'dense' in code_lower and 'concat' in code_lower,
        'is_unet_like': 'unet' in code_lower or ('encoder' in code_lower and 'decoder' in code_lower),
        'is_transformer_like': 'attention' in code_lower or 'transformer' in code_lower,
        'is_mobilenet_like': 'mobile' in code_lower or 'depthwise' in code_lower,
        'is_efficientnet_like': 'efficient' in code_lower or 'mbconv' in code_lower,
        'code_length': len(nn_code),
        'num_classes_defined': nn_code.count('class '),
        'num_functions_defined': nn_code.count('def '),
        'uses_sequential': 'nn.Sequential' in nn_code,
        'uses_modulelist': 'nn.ModuleList' in nn_code,
        'uses_moduledict': 'nn.ModuleDict' in nn_code,
    }


def booleans_to_binary(d):
    for key, value in d.items():
        if isinstance(value, bool):
            d[key] = 1 if value else 0
        elif isinstance(value, dict):
            booleans_to_binary(value)
    return d


def analyze_model_comprehensive(model: nn.Module, nn_code: str, input_tensor) -> dict:
    '''Comprehensive model analysis combining all metrics.'''

    # 1. Basic layer counting
    total_layers = len(list(model.modules()))
    leaf_layers = sum(1 for m in model.modules() if len(list(m.children())) == 0)

    # 2. Parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    # 3. Layer types
    layer_types = {}
    for m in model.modules():
        if len(list(m.children())) == 0:
            layer_type = type(m).__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

    # 4. Depth
    max_depth = get_max_depth(model)        # nn.Module nesting depth
    # Real NN depth (longest weighted path). Guarded a second time so that no
    # future change to get_nn_depth can cost us the rest of the statistics;
    # when it cannot be measured the field is omitted rather than stored as a
    # sentinel value.
    try:
        nn_depth = get_nn_depth(model, input_tensor)
    except Exception:
        nn_depth = None
    depth_info = {} if nn_depth is None else {'nn_depth': nn_depth}

    # 5. Activation functions
    activations = {}
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.Sigmoid,
                          nn.Tanh, nn.ELU, nn.SELU, nn.ReLU6, nn.PReLU)):
            act_type = type(m).__name__
            activations[act_type] = activations.get(act_type, 0) + 1

    # 6. Normalization layers
    norm_types = {}
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                          nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
            norm_type = type(m).__name__
            norm_types[norm_type] = norm_types.get(norm_type, 0) + 1

    # 7. Pooling layers
    pooling_types = {}
    for m in model.modules():
        if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
                          nn.AdaptiveMaxPool2d, nn.MaxPool1d, nn.AvgPool1d)):
            pool_type = type(m).__name__
            pooling_types[pool_type] = pooling_types.get(pool_type, 0) + 1

    # 8. Dropout layers
    dropout_layers = [m for m in model.modules()
                      if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))]
    dropout_count = len(dropout_layers)
    dropout_rates = [m.p for m in dropout_layers]

    # 9. Attention mechanisms
    has_attention = any(isinstance(m, (nn.MultiheadAttention,))
                        for m in model.modules())

    # 10. Convolutional layer details
    conv_info = analyze_conv_layers(model)

    # 11. Linear layer details
    linear_info = analyze_linear_layers(model)

    # 12. Residual connections
    has_residual = has_residual_connections(model)

    # 13. Computational characteristics
    compute_info = analyze_compute_characteristics(model, input_tensor)

    # 14. Architecture patterns
    pattern_info = detect_architecture_patterns(model, nn_code)

    # 15. Parameter distribution by layer type
    param_distribution = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_type = type(module).__name__
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                if layer_type not in param_distribution:
                    param_distribution[layer_type] = 0
                param_distribution[layer_type] += params

    return booleans_to_binary(({'total_layers': total_layers,
                                'leaf_layers': leaf_layers,
                                'max_depth': max_depth,
                                } | depth_info | {
                                'total_params': total_params,
                                'trainable_params': trainable_params,
                                'frozen_params': frozen_params
                                } | compute_info | {'dropout_count': dropout_count,
                                                    'has_attention': has_attention,
                                                    'has_residual_connections': has_residual}
                               | pattern_info | {'meta': {
                # Convolutional details
                'conv_info': conv_info,
                # Linear/Dense details
                'linear_info': linear_info,
                'dropout_rates': dropout_rates,
                'layer_types': layer_types | {'count': {
                    # Specialized layers
                    'activation': activations,
                    'normalization': norm_types,
                    'pooling': pooling_types}},
                'param_distribution': param_distribution}}))

def code_minhash_signature(
    nn_code: str,
    *,
    num_perm: int = 128,
    shingle_n: int = 7,
) -> Dict[str, Any]:
    """
    Return a JSON-serializable MinHash signature for code.
    Stores ONLY hashvalues + metadata (portable + small).
    """
    if not _HAS_DATASKETCH or to_minhash is None:
        return {
            "method": "minhash",
            "available": 0,
            "error": "datasketch not installed / to_minhash not importable",
        }

    mh = to_minhash(nn_code or "", num_perm=num_perm, n=shingle_n)
    # mh.hashvalues is a numpy array; tolist() makes it JSON friendly
    return {
        "method": "minhash",
        "available": 1,
        "num_perm": num_perm,
        "shingle_n": shingle_n,
        "hashvalues": mh.hashvalues.tolist(),
    }

# Read existing JSON
def read_json(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def log_nn_stat(nn_name: Optional[str] = None, max_rows: Optional[int] = None, rewrite: bool = False):
    df = data(nn=nn_name, max_rows=max_rows)
    df = df.drop_duplicates(subset=["nn"], keep="first")

    stat_nn_dir.mkdir(parents=True, exist_ok=True)
    analyzed_nn = {p.stem for p in stat_nn_dir.iterdir() if p.is_file()}

    i = 0
    for _, row in df.iterrows():
        nn = row["nn"]
        f_nm = stat_nn_dir / f"{nn}.json"
        if not (rewrite or nn not in analyzed_nn):
            continue

        prm_id = row["prm_id"]
        i += 1
        print(f"{i}. analyzing NN: {nn}")

        try:
            prm = row["prm"]
            if isinstance(prm, str):
                prm = json.loads(prm.replace("'", '"'))

            local_scope = {"torch": torch, "nn": torch.nn}
            nn_code = row["nn_code"]
            exec(nn_code, local_scope, local_scope)

            out_shape, _, train_set, _ = load_dataset(row["task"], row["dataset"], prm["transform"])
            input_tensor = first_tensor(train_set)
            in_shape = get_in_shape(train_set)

            model = local_scope["Net"](in_shape, out_shape, prm, torch_device()).to(torch_device())

            stats = analyze_model_comprehensive(model, nn_code, input_tensor)

            try:
                stats["code_minhash"] = code_minhash_signature(nn_code, num_perm=128, shingle_n=7)
            except Exception as e:
                stats["code_minhash"] = {"available": 0, "error": repr(e)}

            stats["prm_id"] = prm_id

            with open(f_nm, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4, ensure_ascii=False)

            analyzed_nn.add(nn)

        except Exception as e:
            with open(f_nm, "w", encoding="utf-8") as f:
                json.dump({"prm_id": prm_id, "error": repr(e)}, f, indent=4, ensure_ascii=False)
            analyzed_nn.add(nn)
