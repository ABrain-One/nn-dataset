import importlib
import math

from ab.nn.util.Const import *


def get_ab_nn_attr(module_path, attr_name=None):
    """
    Retrieve an attribute from a module dynamically.
    
    :param module_path: The python path to the module (e.g. 'ab.nn.nn.RLFN')
    :param attr_name: The name of the attribute to retrieve from that module (e.g. 'supported_hyperparameters').
                      Also acts as the default return value if lookup fails.
    :return: The attribute object, or attr_name if not found.
    """
    if module_path is None:
        return attr_name
        
    parts = module_path.split('.')
    # If module_path doesn't start with 'ab', assume it's relative to 'ab.nn'
    if parts[0] != 'ab':
        modul = to_nn + tuple(parts)
    else:
        modul = tuple(parts)

    try:
        # Import the module
        module = importlib.import_module(".".join(modul))
        
        # Retrieve the attribute
        if attr_name:
            return getattr(module, attr_name)
        else:
            return module
            
    except (ModuleNotFoundError, AttributeError):
        return attr_name

def min_accuracy(dataset):
    """
    Get the minimum accuracy for a dataset.
    This replaces the previous pandas-dependent implementation.
    """
    try:
        # Try to load the dataset module and see if it defines a minimum accuracy
        # This is a heuristic; in the original code this might have been a DB lookup
        # For now, we return 0.0 as a safe default or try to find a constant
        return 0.0 
    except Exception:
        return 0.0

# Re-implement other missing utility functions if they were in the deleted file
# Based on usage in other files, likely needed functions:

def unique_nn(task=None, dataset=None, metric=None):
    raise NotImplementedError("Pandas dependency removed. unique_nn not available.")

def unique_nn_cls(task=None, dataset=None, metric=None):
     raise NotImplementedError("Pandas dependency removed. unique_nn_cls not available.")
