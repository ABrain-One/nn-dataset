import os
import json
import optuna
import torch

from ab.nn.api import data
from ab.nn.util.Exception import *
from ab.nn.util import NNAnalysis, Const
from ab.nn.util.Train import optuna_objective
from ab.nn.util.Util import *
from ab.nn.util.db.Calc import patterns_to_configs
from ab.nn.util.db.Read import remaining_trials
from types import MappingProxyType


DATASET_CONFIG = {
    'cifar-10': {'in_shape': (1, 3, 32, 32), 'out_shape': (10,)},
    'cifar-100': {'in_shape': (1, 3, 32, 32), 'out_shape': (100,)},
    'mnist': {'in_shape': (1, 1, 28, 28), 'out_shape': (10,)},
    'fashion-mnist': {'in_shape': (1, 1, 28, 28), 'out_shape': (10,)},
    'svhn': {'in_shape': (1, 3, 32, 32), 'out_shape': (10,)},
    'utkface': {'in_shape': (1, 3, 200, 200), 'out_shape': (1,)},
    'coco': {'in_shape': (1, 3, 224, 224), 'out_shape': (1000,)},
}

def calculate_and_save_stats(nn_name: str, dataset_name: str):
    try:
        df = data(dataset=dataset_name, nn=nn_name)
        if df is None or df.empty: return
        
        if not hasattr(Const, 'stat_dir'): return
        os.makedirs(Const.stat_dir, exist_ok=True)
        
        config = DATASET_CONFIG.get(dataset_name.lower())
        if not config: return
        
        for _, row in df.iterrows():
            prm_id = row.get('prm_id')
            if not prm_id or os.path.exists(os.path.join(Const.stat_dir, f"{prm_id}.json")): continue
            
            try:
                prm = row.get('prm')
                if isinstance(prm, str): prm = json.loads(prm.replace("'", '"'))
                
                local_scope = {'torch': torch, 'nn': torch.nn}
                exec(row.get('nn_code'), local_scope, local_scope)
                model = local_scope['Net'](config['in_shape'], config['out_shape'], prm, torch.device('cpu'))
                
                stats = NNAnalysis.analyze_model_comprehensive(model, row.get('nn_code'), config['in_shape'])
                stats.update({'nn': row.get('nn'), 'prm_id': prm_id})
                with open(os.path.join(Const.stat_dir, f"{prm_id}.json"), 'w') as f: json.dump(stats, f, indent=4)
            except Exception as e:
                with open(os.path.join(Const.stat_dir, f"{prm_id}.json"), 'w') as f: json.dump({'nn': row.get('nn'), 'prm_id': prm_id, 'error': repr(e)}, f, indent=4)
    except: pass

def main(config: str | tuple | list = default_config, nn_prm: dict = default_nn_hyperparameters, epoch_max: int = default_epochs, n_optuna_trials: int | str = default_trials,
         min_batch_binary_power: int = default_min_batch_power, max_batch_binary_power: int = default_max_batch_power,
         min_learning_rate: float = default_min_lr, max_learning_rate: float = default_max_lr,
         min_momentum: float = default_min_momentum, max_momentum: float = default_max_momentum,
         min_dropout: float = default_min_dropout, max_dropout: float = default_max_dropout,
         transform: str | tuple = None, nn_fail_attempts: int = default_nn_fail_attempts, random_config_order: bool = default_random_config_order,
         num_workers: int = default_num_workers, pretrained: int = default_pretrained, epoch_limit_minutes: int = default_epoch_limit_minutes,
         train_missing_pipelines: bool = default_train_missing_pipelines, save_pth_weights: bool = default_save_pth_weights, save_onnx_weights: int = default_save_onnx_weights):
    """
    Main function for training models using Optuna optimization.

    NN pipeline configuration examples
    conf = ''  # For all configurations
    conf = 'img-classification' # For all image classification configurations
    conf = 'img-classification_cifar-10_acc' # For a particular configuration for all models for CIFAR-10
    conf = 'img-classification_cifar-10_acc_GoogLeNet'  # For a particular configuration and model for CIFAR-10
    conf = 'img-classification_mnist_acc' # For a particular configuration for all models for MNIST
    conf = 'img-classification_mnist_acc_GoogLeNet'  # For a particular configuration and model for MNIST
    conf = ('img-classification', 'img-segmentation')  # For all image classification and segmentation configurations

    :param config: Configuration specifying the model training pipelines. The default value for all configurations.
    :param nn_prm: Fixed hyperparameter values for neural network training, e.g. {"lr": 0.0061, "momentum": 0.7549, "batch": 4}.
    :param epoch_max: Number of training epochs.
    :param n_optuna_trials: The total number of Optuna trials the model should have. If negative, its absolute value represents the number of additional trials.
    :param min_batch_binary_power: Minimum power of two for batch size. E.g., with a value of 0, batch size equals 2**0 = 1.
    :param max_batch_binary_power: Maximum power of two for batch size. E.g., with a value of 12, batch size equals 2**12 = 4096.
    :param min_learning_rate: Minimum value of learning rate.
    :param max_learning_rate: Maximum value of learning rate.
    :param min_momentum: Minimum value of momentum.
    :param max_momentum: Maximum value of momentum.
    :param min_dropout: Minimum value of dropout.
    :param max_dropout: Maximum value of dropout.
    :param transform: Transformation algorithm name. If None (default), all available algorithms are used by Optuna.
    :param nn_fail_attempts: Number of attempts if the neural network model throws exceptions.
    :param random_config_order: If random shuffling of the config list is required.
    :param num_workers: Number of data loader workers.
    :param pretrained: Control use of NN pretrained weights: 1 (always use), 0 (never use), or default (let Optuna decide).
    :param epoch_limit_minutes: Maximum duration per training epoch, minutes.
    :param train_missing_pipelines: Find and train all missing training pipelines for provided configuration.
    :param save_pth_weights: Enable saving of the best model weights in PyTorch checkpoints.
    :param save_onnx_weights: Save the best model in ONNX format: 1 (save), 0 (don't save).
    """

    validate_prm(min_batch_binary_power, max_batch_binary_power, min_learning_rate, max_learning_rate, min_momentum, max_momentum, min_dropout, max_dropout)
    nn_prm = MappingProxyType(nn_prm)

    # Determine configurations based on the provided config
    sub_configs = patterns_to_configs(config, random_config_order, train_missing_pipelines)
    if transform:
        transform = transform if isinstance(transform, (tuple, list)) else (transform,)
    print(f"Training configurations ({epoch_max} epochs):")
    for idx, sub_config in enumerate(sub_configs, start=1):
        print(f"{idx}. {sub_config}")
    for sub_config in sub_configs:
        sub_config_ext = sub_config + (epoch_max,)
        n_optuna_trials_left, n_passed_trials = remaining_trials(sub_config_ext, n_optuna_trials)
        n_expected_trials = n_optuna_trials_left + n_passed_trials

        conf_str = ', '.join([f"{n}: {v}" for n, v in zip(main_columns_ext, sub_config_ext)])
        if n_optuna_trials_left == 0:
            print("All trials have already been passed for the " + conf_str)
        else:
            print("\nStarting training for the " + conf_str)
            fail_iterations = nn_fail_attempts
            continue_study = True
            max_batch_binary_power_local = max_batch_binary_power
            _, dataset, _, nn = sub_config
            last_accuracy = None
            while (continue_study and max_batch_binary_power_local >= min_batch_binary_power and fail_iterations > -1
                   and remaining_trials(sub_config_ext, n_expected_trials)[0] > 0):
                continue_study = False
                try:
                    # Launch Optuna for the current NN model
                    study = optuna.create_study(study_name=nn, direction='maximize')

                    # Configure Optuna for the current model
                    def objective(trial):
                        nonlocal continue_study, fail_iterations, max_batch_binary_power_local, last_accuracy
                        try:
                            accuracy, accuracy_to_time, duration = optuna_objective(trial, sub_config, nn_prm, num_workers, min_learning_rate, max_learning_rate,
                                                                                    min_momentum, max_momentum, min_dropout, max_dropout,
                                                                                    min_batch_binary_power, max_batch_binary_power_local, transform, fail_iterations, epoch_max,
                                                                                    pretrained, epoch_limit_minutes, save_pth_weights, save_onnx_weights)
                            # calculate_and_save_stats(nn, dataset) # todo: uncomment temporary commented functionality
                            if good(accuracy, min_accuracy(dataset), duration):
                                fail_iterations = nn_fail_attempts
                            last_accuracy = accuracy
                            return accuracy
                        except Exception as e:
                            print(f"Optuna: exception in objective function for nn {nn}: {e}")
                            if fail_iterations > -1: continue_study = True
                            if isinstance(e, CudaOutOfMemory):
                                raise e
                            fail_iterations -= 1
                            if fail_iterations <= 0:
                                raise e
                            return 0.0

                    study.optimize(objective, n_trials=n_optuna_trials_left)
                    return last_accuracy
                except CudaOutOfMemory as e:
                    max_batch_binary_power_local = e.batch_size_power() - 1
                    print(f"Max batch is decreased to {max_batch(max_batch_binary_power_local)} due to a CUDA Out of Memory Exception for model '{nn}'")
                except:
                    pass
                finally:
                    del study
                    release_memory()


if __name__ == "__main__":
    a = args()
    main(a.config, a.nn_prm, a.epochs, a.trials, a.min_batch_binary_power, a.max_batch_binary_power,
         a.min_learning_rate, a.max_learning_rate, a.min_momentum, a.max_momentum, a.min_dropout, a.max_dropout, a.transform,
         a.nn_fail_attempts, a.random_config_order, a.workers, a.pretrained, a.epoch_limit_minutes, a.train_missing_pipelines,
         a.save_pth_weights, a.save_onnx_weights)
