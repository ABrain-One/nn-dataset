from ab.nn import train
from ab.nn.util.Util import args

if __name__ == "__main__":
    a = args()

    # NN pipeline configuration examples
    conf = a.config  # From the command line argument --config
    # conf = ''  # For all configurations
    # conf = 'img-classification' # For all image classification configurations
    # conf = 'img-classification_cifar-10_acc' # For a particular configuration for all models for CIFAR-10
    # conf = 'img-classification_cifar-10_acc_GoogLeNet'  # For a particular configuration and model for CIFAR-10
    # conf = 'img-classification_mnist_acc' # For a particular configuration for all models for MNIST
    # conf = 'img-classification_mnist_acc_GoogLeNet'  # For a particular configuration and model for MNIST
    # conf = ('img-classification', 'img-segmentation')  # For all image classification and segmentation configurations

    # Number of epochs.
    epochs = a.epochs

    # Number of Optuna trial examples.
    trials = a.trials  # From the command line argument --trials
    # trials = 100  # 100 trials
    # trials = -1  # Try once more: for quick verification of model training process after code modifications.
    # trials = -5  # Try 5 more times: to thoroughly verify model training after code modifications.

    # Other command line arguments
    min_batch_power = a.min_batch_binary_power
    max_batch_power = a.max_batch_binary_power
    min_lr = a.min_learning_rate
    max_lr = a.max_learning_rate
    min_mom = a.min_momentum
    max_mom = a.max_momentum
    transform = a.transform
    nn_fail_attempts = a.nn_fail_attempts
    random_config_order = a.random_config_order

    ''' !!! Please commit correct statistics whenever it's generated !!! '''
    # Run training with Optuna: detects and saves performance metric values
    train.main(conf, epochs, trials, min_batch_power, max_batch_power, min_lr, max_lr, min_mom, max_mom, transform, nn_fail_attempts, random_config_order)

