import os
import importlib
import importlib.util

def get_obj(name, o_type):
    """ Dynamically load a function/field by name if provided from the object of type 'o_type'"""
    if o_type == 'loader':
        module = importlib.import_module(f"ab.nn.loader.{name}")
        return getattr(module, o_type)

    from ab.nn.util.db.Util import get_ab_nn_attr
    return get_ab_nn_attr(f"{o_type}.{name}", o_type)



def load_dataset(task, dataset_name, transform_name, transform_dir=None, test_only=False):
    """
    Dynamically load dataset and transformation based on the provided paths.
    :param task: Task name
    :param dataset_name: Dataset name
    :param transform_name: Transform name
    :param transform_dir: Optional directory to load transform
    :param test_only: False (default): for loaders with a true test split (Split.TRUE_TEST_LOADERS) a validation set
        is carved out of the training split by ab.nn.util.Split (seeded, stratified, test transform; test set
        unchanged): train_set is the remaining part and the validation set is the attribute val_set of the result.
        Other datasets keep their train/test sets (val_set None, one log line). True: the loader's result as is.
    :return: (out_shape, minimum_accuracy, train_set, test_set); by default a Split.LoadedDataset, a 4-tuple that
        also carries val_set.
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

    if test_only:
        return loader(transform, task)
    from ab.nn.util import Split
    return Split.with_validation(dataset_name, loader(transform, task))
