from ab.nn.util.Util import get_ab_nn_attr
import os
import importlib.util

def get_obj(name, o_type):
    """ Dynamically load a function/field by name if provided from the object of type 'o_type'"""
    return get_ab_nn_attr(f"{o_type}.{name}", o_type)



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

    # Check if custom directory is provided and file exists
    if transform_dir and transform_name:
        transform_file_path = os.path.join(transform_dir, f"{transform_name}.py")
        if os.path.exists(transform_file_path):

            # Import the transform function 
            spec = importlib.util.spec_from_file_location(transform_name, transform_file_path)
            transform_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(transform_module)
            
            # Check 'transform' function in the module
            if hasattr(transform_module, 'transform'):
                transform = transform_module.transform
            else:
                # Fallback to dynamic loading if no transform function found
                transform = get_obj(transform_name, 'transform')
        else:
            # Fallback to dynamic loading if file doesn't exist
            transform = get_obj(transform_name, 'transform')
    else:
        # Use original behavior if no custom directory
        transform = get_obj(transform_name, 'transform')

    return loader(transform, task)