import json
from os import listdir, makedirs
import os
from ab.nn.util.Util import *
from ab.nn.util.db.Write import init_population

init_population()


def data(only_best_accuracy=False, task=None, dataset=None, metric=None, nn=None, epoch=None) -> tuple[dict[str, int | float | str | dict[str, int | float | str]], ...]:
    # todo: update this function to load data from dataset
    """
    Get the NN model code and all related statistics
    :param only_best_accuracy: if True returns for every NN model only statistics for the best accuracy, else all collected statistics
    :param task: Name of the task. If provided, it is used to filter the data.
    :param dataset: Name of dataset. If provided, it is used to filter the data.If provided, it is used to filter the data.
    :param metric: Name of the NN performance metric. If provided, it is used to filter the data.
    :param nn: Neural network name. If provided, it is used to filter the data.
    :param epoch: Epoch of the NN training process. If provided, it is used to filter the data.

    Returns: Tuple of dictionaries. Each dictionary has the following keys with types
                                  {'task': str, 'dataset': str,
                                   'metric': str, 'metric_code': str,
                                   'nn': str, 'nn_code': str,
                                   'epoch': int, 'accuracy': float,
                                   'duration': int,
                                   'prm': dict,
                                   'transform_code': <transformer code: str>}
    where
    - duration is duration of the training in milliseconds
    - prm is dict[str, int | float | str] of NN training hyperparameters including, but not limiting to
    following keys:
    {
        'batch': 4,
        'dropout': 0.17920158482473114,
        'lr': 0.02487720458587122,
        'momentum': 0.3867297180491852,
        'transform': 'norm_299'
    }
    """

    out = []
    for stat_folder in listdir(stat_dir):
        curr_task, curr_dataset, curr_metric, nn = conf_to_names(stat_folder)

        with open(metric_dir / (curr_metric + '.py'), 'r') as code_file:
            metric_code = str(code_file.read())

        with open(nn_dir / (nn + '.py'), 'r') as code_file:
            nn_code = str(code_file.read())

        for epoch_file in listdir(stat_dir / stat_folder):
            with open(stat_dir / stat_folder / epoch_file, 'r') as json_file:
                content = json.load(json_file)

            for stat in content:
                with open(transform_dir / (stat['transform'] + '.py'), 'r') as code_file:
                    transform_code = str(code_file.read())
                stat.pop('uid')
                next_row = {
                    'task': curr_task,
                    'dataset': curr_dataset,
                    'metric': curr_metric,
                    'metric_code': metric_code,
                    'nn': nn,
                    'nn_code': nn_code,
                    'epoch': Path(epoch_file).stem,
                    'accuracy': stat.pop('accuracy'),
                    'duration': stat.pop('duration', None),
                    'prm': stat,
                    'transform_code': transform_code
                }
                out.append(next_row)
                if only_best_accuracy: break
    return tuple(out)


def remaining_trials(trial_file, model_name, n_optuna_trials):
    """
    Calculates the remaining Optuna trials based on the completed ones. Checks for a 'trials.json' file in the
    specified directory to determine how many trials have been completed, and returns the number of trials left.
    :param trial_file: Trial file path
    :param model_name: Name of the model.
    :param n_optuna_trials: The total number of Optuna trials the model should have. If negative, its absolute value represents the number of additional trials.
    :return: Number of remaining trials.
    """
    n_passed_trials = 0
    if exists(trial_file):
        with open(trial_file, 'r') as f:
            trials = json.load(f)
            n_passed_trials = len(trials)
    n_remaining_trials = abs(n_optuna_trials) if n_optuna_trials < 0 else max(0, n_optuna_trials - n_passed_trials)
    if n_passed_trials > 0:
        print(f"The {model_name} passed {n_passed_trials} trials, {n_remaining_trials} left.")
    return n_remaining_trials


def supported_transformers():
    return ['complex','complex_32','complex_64','complex_128','complex_256','complex_299', 'complex_512',
            'norm', 'norm_32', 'norm_64', 'norm_128', 'norm_256', 'norm_299', 'norm_512',
            'echo', 'echo_32', 'echo_64', 'echo_128', 'echo_256', 'echo_299', 'echo_512']


# todo: Request from the database unique names of all configures corresponding to config-patterns
# once the database is loaded, the function will be updated

def unique_configs(patterns: tuple[str, ...]) -> list[str]:
    """
    Collect models matching the given configuration prefix.
    :param patterns: Tuple of configuration prefixes.
    :return: List of unique configuration directories.
    """
    all_configs = []
    for pattern in patterns:
        # Collect configurations matching the pattern
        matching_configs = [
            config for config in os.listdir(stat_dir) if config.startswith(pattern)
        ]
        # If no match, create a directory if the pattern is a full configuration
        if not matching_configs and is_full_config(pattern):
            os.makedirs(os.path.join(stat_dir, pattern), exist_ok=True)
        all_configs.extend(matching_configs)
    return list(set(all_configs))

