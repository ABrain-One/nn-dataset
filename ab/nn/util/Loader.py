from ab.nn.util.Util import get_ab_nn_attr, nn_mod
from ab.nn.util.Const import transform_dir as default_transform_dir
import os
import importlib.util

def get_obj(name, o_type):
    """ Dynamically load a function/field by name if provided from the object of type 'o_type'"""
    try:
        return get_ab_nn_attr(f"{o_type}.{name}", o_type)
    except (ModuleNotFoundError, AttributeError) as e:
        # If standard import fails and it's a transform, try file-based loading
        if o_type == 'transform':
            return load_transform_from_file(name)
        raise e


def load_transform_from_file(transform_name, transform_dir=None):
    """
    Load transform function from a file, handling hyphens in filenames.
    :param transform_name: Name of the transform (may contain hyphens)
    :param transform_dir: Optional directory to load transform from
    :return: Transform function
    """
    if transform_dir is None:
        transform_dir = default_transform_dir
    
    transform_file_path = os.path.join(transform_dir, f"{transform_name}.py")
    
    if not os.path.exists(transform_file_path):
        raise ModuleNotFoundError(f"Transform file not found: {transform_file_path}")
    
    # Use a valid module name (replace hyphens with underscores for the spec)
    module_name = f"transform_{transform_name.replace('-', '_').replace('.', '_')}"
    
    spec = importlib.util.spec_from_file_location(module_name, transform_file_path)
    transform_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transform_module)
    
    # Check for 'transform' function in the module
    if hasattr(transform_module, 'transform'):
        return transform_module.transform
    else:
        raise AttributeError(f"Transform module {transform_name} does not have a 'transform' function")


def load_dataset(task, dataset_name, transform_name, transform_dir=None):
    """
    Dynamically load dataset and transformation based on the provided paths.
    :param task: Task name
    :param dataset_name: Dataset name
    :param transform_name: Transform name
    :param transform_dir: Optional directory to load transform
    :return: Train and test datasets.
    """
    
    loader = get_obj(dataset_name, 'loader')

    # Try to load transform with improved error handling
    try:
        # First, try standard Python import (for transforms without hyphens)
        transform = get_obj(transform_name, 'transform')
    except (ModuleNotFoundError, AttributeError):
        # If that fails, try file-based loading (handles hyphens)
        try:
            transform = load_transform_from_file(transform_name, transform_dir)
            print(f"Loaded transform '{transform_name}' from file (contains hyphens)")
        except Exception as e:
            # If all else fails, use a default transform
            print(f"Warning: Could not load transform '{transform_name}': {e}")
            print(f"Falling back to default transform 'norm_256'")
            try:
                transform = get_obj('norm_256', 'transform')
            except:
                # Ultimate fallback to echo (identity transform)
                print(f"Warning: Could not load 'norm_256', using 'echo' as final fallback")
                transform = get_obj('echo', 'transform')

    return loader(transform, task)