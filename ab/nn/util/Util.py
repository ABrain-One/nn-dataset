import argparse
import datetime
import gc
import hashlib
import importlib.util
import inspect
import random
import re
import json

import torch
from os import makedirs, remove
from os.path import exists, dirname

from ab.nn.util.Const import *


def nn_mod(*nms):
    return ".".join(to_nn + nms)


def create_file(file_dir, file_name, content=''):
    file_path = file_dir / file_name
    if not exists(file_path):
        makedirs(file_dir, exist_ok=True)
    else:
        remove(file_path)
    with open(file_path, 'w') as file:
        file.write(content if content else '')
    return file_path


def get_obj_attr(obj, f_name, default=None):
    return getattr(obj, f_name) if hasattr(obj, f_name) else default


def torch_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def get_attr(mod, f):
    return get_obj_attr(__import__(mod, fromlist=[f]), f)


def get_ab_nn_attr(mod, f):
    return get_attr(nn_mod(mod), f)


def min_accuracy(dataset):
    return get_ab_nn_attr(f"loader.{dataset}", 'MINIMUM_ACCURACY')


def order_configs(configs, random_config_order):
    configs = list(configs)
    if random_config_order:
        random.shuffle(configs)
    else:
        configs.sort()
    return configs


def conf_to_names(c: str) -> tuple[str, ...]:
    return tuple(c.split(config_splitter))


def add_categorical_if_absent(trial, prms, nm, fn, default=None):
    if not (nm in prms and prms[nm]):
        prms[nm] = trial.suggest_categorical(nm, default or fn())
    return prms[nm]

def is_full_config(l: list[str] | tuple[str, ...]):
    return 4 == len(l) and (nn_dir / (l[-1] + '.py')).exists()


def merge_prm(prm: dict, d: dict):
    prm.update(d)
    prm = dict(sorted(prm.items()))
    return prm


def max_batch(binary_power):
    return 2 ** binary_power


def model_stat_dir(config):
    return stat_dir / config_splitter.join(config)


def accuracy_to_time_metric(accuracy, min_accuracy, training_duration) -> float:
    """
    Naive 'accuracy to time' metric for fixed number of training epochs.
    This metric is essential for detecting the fastest accuracy improvements during neural network training.
    """
    if accuracy is None:
        accuracy = 0.0
    if min_accuracy is None:
        min_accuracy = 0.0
    d = max(0.0, (accuracy - min_accuracy)) / (training_duration / 1e11)
    print(f"accuracy_to_time_metric {d}")
    return d


def good(result, minimum_accuracy, duration):
    if minimum_accuracy is None:
        minimum_accuracy = 0.0
    return result > minimum_accuracy * 1.2


def uuid4(obj):
    s = re.sub('\\s', '', str(obj))
    res = hashlib.md5(s.encode())
    return res.hexdigest()


def validate_prm(batch_min, batch_max, lr_min, lr_max, momentum_min, momentum_max, dropout_min, dropout_max):
    if batch_min and batch_max and batch_min > batch_max: raise Exception(f"min_batch_binary_power {batch_min} > max_batch_binary_power {batch_max}")
    if lr_min and lr_max and lr_min > lr_max: raise Exception(f"min_learning_rate {lr_min} > max_learning_rate {lr_max}")
    if momentum_min and momentum_max and momentum_min > momentum_max: raise Exception(f"min_momentum {momentum_min} > max_momentum {momentum_max}")
    if dropout_min and dropout_max and dropout_min > dropout_max: raise Exception(f"min_momentum {dropout_min} > max_momentum {dropout_max}")


def format_time(sec):
    return datetime.timedelta(seconds=int(sec))


def release_memory():
    try:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e:
        print(f"Exception during memory release: {e}")


def read_py_file_as_string(file_path):
    """
    read_py_file_as_string。

    param:
        file_path (str): path of the file to read.

    Return:
        str: Content of the file.
    """
    try:
        spec = importlib.util.spec_from_file_location("module_name", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        source_code = inspect.getsource(module)
        return source_code
    except Exception as e:
        print(f"error when reading file: {e}")
        return None


def str_not_none(prefix, value):
    if value:
        return ' ' + prefix + str(value)
    else:
        return ''


def export_model_to_onnx(model, dummy_input, path):
    model.eval()
    assert isinstance(model, torch.nn.Module)
    hasAdaptivePoolingLayer = False
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveMaxPool2d)):
            if layer.output_size not in [(1, 1), 1, None]:
                hasAdaptivePoolingLayer = True
    # Ensure if the directory exists
    onnx_directory = dirname(path)
    makedirs(onnx_directory, exist_ok=True)
    with torch.no_grad():
        if hasAdaptivePoolingLayer:
            torch.onnx.export(
                model,
                dummy_input,
                path,
                input_names=["input"],
                output_names=["output"])
        else:
            torch.onnx.export(
                model,
                dummy_input,
                path,  # Use the provided path
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size", 2: "height", 3: "width"},
                    "output": {0: "batch_size"}})
    print(f"Exported neural network to ONNX format at {path}")

#  FUNCTIONS FOR SAVING AND LOADING WEIGHTS

def export_torch_weights(model, path):
    """
    Saves the trained weights of a model's state_dict to the specified path.
    This is a general function that can be used for any PyTorch model.
    """
    # Ensure the directory exists
    makedirs(dirname(path), exist_ok=True)
    print(f"Exporting model weights to {path}...")
    torch.save(model.state_dict(), path)
    print(f"Export complete. Weights saved to {path}")


def load_torch_weights(model, path):
    """
    Loads trained weights from a .pth file into a model instance.
    This is a general function that can be used for any PyTorch model.
    """
    device = next(model.parameters()).device
    if os.path.exists(path):
        print(f"Loading weights from {path}...")
        model.load_state_dict(torch.load(path, map_location=device))
        print("Weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Weights file not found at {path}. Cannot load model.")


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=default_config,
                        help="Configuration specifying the model training pipelines. The default value for all configurations.")
    parser.add_argument('-p', '--nn_prm', type=json.loads, default=default_nn_hyperparameters,
                        help="JSON string with fixed hyperparameter values for neural network training, e.g. -p '{\"lr\": 0.0061, \"momentum\": 0.7549, \"batch\": 4}'")
    parser.add_argument('-e', '--epochs', type=int, default=default_epochs,
                        help="Numbers of training epochs.")
    parser.add_argument('-t', '--trials', type=int, default=default_trials,
                        help="The total number of Optuna trials the model should have. If negative, its absolute value represents the number of additional trials.")
    parser.add_argument('--min_batch_binary_power', type=int, default=default_min_batch_power,
                        help="Minimum power of two for batch size. E.g., with a value of 0, batch size equals 2**0 = 1.")
    parser.add_argument('-b', '--max_batch_binary_power', type=int, default=default_max_batch_power,
                        help="Maximum power of two for batch size. E.g., with a value of 12, batch size equals 2**12 = 4096.")
    parser.add_argument('--min_learning_rate', type=float, default=default_min_lr,
                        help="Minimum value of learning rate.")
    parser.add_argument('-l', '--max_learning_rate', type=float, default=default_max_lr,
                        help="Maximum value of learning rate.")
    parser.add_argument('--min_momentum', type=float, default=default_min_momentum,
                        help="Minimum value of momentum.")
    parser.add_argument('-m', '--max_momentum', type=float, default=default_max_momentum,
                        help="Maximum value of momentum.")

    parser.add_argument('--min_dropout', type=float, default=default_min_dropout,
                        help="Minimum value of dropout.")
    parser.add_argument('-d', '--max_dropout', type=float, default=default_max_dropout,
                        help="Maximum value of dropout.")

    parser.add_argument('-f', '--transform', type=str, default=default_transform,
                        help="The transformation algorithm name. If None (default), all available algorithms are used by Optuna.")
    parser.add_argument('-a', '--nn_fail_attempts', type=int, default=default_nn_fail_attempts,
                        help="Number of attempts if the neural network model throws exceptions.")
    parser.add_argument('-r', '--random_config_order', type=bool, default=default_random_config_order,
                        help=f"If set to True, randomly shuffles the configuration list. Default is {default_random_config_order}.")
    parser.add_argument('-w', '--workers', type=int, default=default_num_workers,
                        help="Number of data loader workers.")
    parser.add_argument('--pretrained', type=int, choices=[1, 0], default=default_pretrained,
                        help='Control pretrained weights usage: 1 (always use), 0 (never use), or default (let Optuna decide)')
    parser.add_argument('--save_onnx_weights', type=int, choices=[1, 0], default=default_save_onnx_weights,
                        help='Save the best model in ONNX format: 1 (save), 0 (don\'t save)')
    parser.add_argument('--epoch_limit_minutes', type=int, default=default_epoch_limit_minutes,
                        help=f'Maximum duration per training epoch, minutes; default {default_epoch_limit_minutes} minutes')
    parser.add_argument('--train_missing_pipelines', type=bool, default=default_train_missing_pipelines,
                        help=f'Find and train all missing training pipelines for the provided configuration; default {default_train_missing_pipelines}')
    return parser.parse_args()
