import json
import torch
import torch.nn as nn
import time
from collections import defaultdict


IN_SHAPE = (1, 3, 224, 224)   # Batch size 1, 3 channels, 224x224 image
OUT_SHAPE = (10,)              # 10 classes


def get_max_depth(module, depth=0):
    """Calculate maximum depth of the model."""
    children = list(module.children())
    if not children:
        return depth
    return max(get_max_depth(child, depth + 1) for child in children)


def analyze_conv_layers(model):
    """Analyze convolutional layers."""
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
        "count": len(conv_layers),
        "kernel_sizes": kernel_sizes,
        "strides": strides,
        "padding_values": padding_values,
        "avg_kernel_size": sum(kernel_sizes) / len(kernel_sizes) if kernel_sizes else 0,
        "avg_stride": sum(strides) / len(strides) if strides else 0,
        "total_conv_params": sum(p.numel() for m in conv_layers for p in m.parameters())
    }


def analyze_linear_layers(model):
    """Analyze linear/dense layers."""
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    
    if not linear_layers:
        return {}
    
    return {
        "count": len(linear_layers),
        "input_dims": [m.in_features for m in linear_layers],
        "output_dims": [m.out_features for m in linear_layers],
        "total_linear_params": sum(p.numel() for m in linear_layers 
                                  for p in m.parameters()),
        "has_bias": [m.bias is not None for m in linear_layers]
    }


def has_residual_connections(model):
    """Detect if model has residual/skip connections."""
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
    """Rough FLOPs estimation for common layers."""
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


def analyze_compute_characteristics(model: nn.Module, input_shape: tuple) -> dict:
    """Analyze computational requirements."""
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # FLOPs estimation
    flops = estimate_flops(model, dummy_input)
    
    # Memory footprint
    model_size_mb = sum(p.numel() * p.element_size() 
                        for p in model.parameters()) / (1024 ** 2)
    
    # Calculate buffer memory (e.g., BatchNorm running stats)
    buffer_size_mb = sum(b.numel() * b.element_size() 
                         for b in model.buffers()) / (1024 ** 2)
    
    # Inference time (rough estimate)
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
        
        # Measure
        start = time.time()
        num_runs = 100
        for _ in range(num_runs):
            _ = model(dummy_input)
        avg_inference_time = (time.time() - start) / num_runs
    
    return {
        "estimated_flops": flops,
        "estimated_gflops": flops / 1e9,
        "model_size_mb": model_size_mb,
        "buffer_size_mb": buffer_size_mb,
        "total_memory_mb": model_size_mb + buffer_size_mb,
        "avg_inference_time_ms": avg_inference_time * 1000
    }


def detect_architecture_patterns(model: nn.Module, nn_code: str) -> dict:
    """Detect high-level architecture patterns."""
    code_lower = nn_code.lower()
    
    return {
        "is_resnet_like": 'residual' in code_lower or 'resnet' in code_lower,
        "is_vgg_like": 'vgg' in code_lower,
        "is_inception_like": 'inception' in code_lower,
        "is_densenet_like": 'dense' in code_lower and 'concat' in code_lower,
        "is_unet_like": 'unet' in code_lower or ('encoder' in code_lower and 'decoder' in code_lower),
        "is_transformer_like": 'attention' in code_lower or 'transformer' in code_lower,
        "is_mobilenet_like": 'mobile' in code_lower or 'depthwise' in code_lower,
        "is_efficientnet_like": 'efficient' in code_lower or 'mbconv' in code_lower,
        "code_length": len(nn_code),
        "num_classes_defined": nn_code.count('class '),
        "num_functions_defined": nn_code.count('def '),
        "uses_sequential": 'nn.Sequential' in nn_code,
        "uses_modulelist": 'nn.ModuleList' in nn_code,
        "uses_moduledict": 'nn.ModuleDict' in nn_code,
    }


def analyze_model_comprehensive(model: nn.Module, nn_code: str, input_shape: tuple) -> dict:
    """Comprehensive model analysis combining all metrics."""
    
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
    activation_types = {}
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.Sigmoid, 
                         nn.Tanh, nn.ELU, nn.SELU, nn.ReLU6, nn.PReLU)):
            act_type = type(m).__name__
            activation_types[act_type] = activation_types.get(act_type, 0) + 1
    
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
    compute_info = analyze_compute_characteristics(model, input_shape)
    
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
    
    return {
        # Basic structure
        "total_layers": total_layers,
        "leaf_layers": leaf_layers,
        "max_depth": max_depth,
        "layer_types": layer_types,
        
        # Parameters
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "param_distribution": param_distribution,
        
        # Activations and normalization
        "activation_types": activation_types,
        "normalization_types": norm_types,
        
        # Specialized layers
        "pooling_types": pooling_types,
        "dropout_count": dropout_count,
        "dropout_rates": dropout_rates,
        "has_attention": has_attention,
        "has_residual_connections": has_residual,
        
        # Convolutional details
        "conv_info": conv_info,
        
        # Linear/Dense details
        "linear_info": linear_info,
        
        # Computational characteristics
        "compute_characteristics": compute_info,
        
        # Architecture patterns
        "architecture_patterns": pattern_info
    }


def main():
    # Import your data function
    from ab.nn.api import data  # Replace with the module where your data() function is
    
    df = data()
    df_selected = df[['nn', 'nn_code', 'prm']]  # first 10 models
    
    results = []
    
    for idx, row in df_selected.iterrows():
        nn_name = row['nn']
        nn_code = row['nn_code']
        prm = row['prm']
        
        print(f"Analyzing model {idx + 1}/{len(df_selected)}: {nn_name}")
        
        try:
            # Execute the code to define all classes in a local namespace
            local_ns = {"torch": torch, "nn": nn}
            exec(nn_code, local_ns, local_ns)
            
            # Instantiate the model (assumes main class is called 'Net')
            device = torch.device('cpu')
            model = local_ns['Net'](in_shape=IN_SHAPE, out_shape=OUT_SHAPE, prm=prm, device=device)
            
            # Comprehensive analysis
            characteristics = analyze_model_comprehensive(model, nn_code, IN_SHAPE)
            
            results.append({
                "nn": nn_name,
                "prm": prm,
                "characteristics": characteristics
            })
            
            

            print(f"  ✓ Successfully analyzed {nn_name}")
            print(f"    Total params: {characteristics['total_params']:,}")
            print(f"    GFLOPs: {characteristics['compute_characteristics']['estimated_gflops']:.2f}")
            print(f"    Model size: {characteristics['compute_characteristics']['model_size_mb']:.2f} MB")
            
        except Exception as e:
            print(f"  ✗ Failed to analyze model {nn_name}: {e}")
            results.append({
                "nn": nn_name,
                "prm": prm,
                "error": str(e)
            })
    
    # Save the characteristics to JSON
    output_file = "nn_characteristics_comprehensive.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total models analyzed: {len(results)}")
    print(f"Successful analyses: {sum(1 for r in results if 'error' not in r)}")
    print(f"Failed analyses: {sum(1 for r in results if 'error' in r)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()