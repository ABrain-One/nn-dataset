import torchvision
import torchvision.transforms as transforms

from ab.nn.util.Const import data_dir

# MNIST-specific normalization values
__norm_mean = (0.5,)
__norm_dev = (0.5,)

# MNIST class-related constants
__class_quantity = 10
__minimum_accuracy = 1.0 / __class_quantity

def loader(transform_fn):
    """
    Loader function for the MNIST dataset.
    :param transform_fn: Function to apply transformations.
    :return: Tuple containing class quantity, minimum accuracy, train set, and test set.
    """
    # Apply transformations using provided function
    transform = transform_fn((__norm_mean, __norm_dev))

    # Load MNIST train and test sets
    train_set = torchvision.datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

    return (__class_quantity,), __minimum_accuracy, train_set, test_set
