import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ab.nn.api import data
from ab.nn.util.Const import stat_nn_dir
from ab.nn.util.Loader import load_dataset
from ab.nn.util.Util import get_in_shape, torch_device, first_tensor
from ab.nn.util.db.Write import save_nn_stat


def get_max_depth(module, depth=0):
    '''Calculate maximum depth of the model.'''
    children = list(module.children())
    if not children:
        return depth
    return max(get_max_depth(child, depth + 1) for child in children)


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
    max_depth = get_max_depth(model)

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


# Read existing JSON
def read_json(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def log_nn_stat(nn_name: str, max_rows: Optional[int] = None, rewrite: bool = False, save_to_db: bool = True):
    try:
        df = data(nn=nn_name, max_rows=max_rows, unique_nn=True)

        stat_nn_dir.mkdir(parents=True, exist_ok=True)
        analyzed_nn = set([p.stem for p in stat_nn_dir.iterdir() if p.is_file()])

        i = 0
        for _, row in df.iterrows():
            nn_name = row['nn']
            f_nm = stat_nn_dir / f'{nn_name}.json'
            if rewrite or not nn_name in analyzed_nn:
                prm_id = row['prm_id']
                i += 1
                print(f'{i}. analyzing NN: {nn_name}')
                try:
                    prm = row['prm']
                    if isinstance(prm, str): prm = json.loads(prm.replace("'", '"'))
                    local_scope = {'torch': torch, 'nn': torch.nn}
                    nn_code = row['nn_code']
                    exec(nn_code, local_scope, local_scope)
                    out_shape, _, train_set, _ = load_dataset(row['task'], row['dataset'], prm['transform'])
                    input_tensor = first_tensor(train_set)
                    in_shape = get_in_shape(train_set)
                    model = local_scope['Net'](in_shape, out_shape, prm, torch_device())
                    model.to(torch_device())
                    stats = analyze_model_comprehensive(model, nn_code, input_tensor)
                    stats.update({'prm_id': prm_id})

                    # Save to JSON file
                    with open(f_nm, 'w') as f:
                        json.dump(stats, f, indent=4)

                    # Save to database if requested
                    if save_to_db:
                        save_nn_stat(nn_name, prm_id, stats)

                except Exception as e:
                    error_stats = {'prm_id': prm_id, 'error': repr(e)}

                    # Save error to JSON file
                    with open(f_nm, 'w') as f:
                        json.dump(error_stats, f, indent=4)

                    # Save error to database if requested
                    if save_to_db:
                        save_nn_stat(nn_name, prm_id, error_stats)

    except Exception as e:
        print(e)
        pass
