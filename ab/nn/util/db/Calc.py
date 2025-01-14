import json
import random
from os.path import exists

import ab.nn.util.db.Write as DB_Write
import ab.nn.util.db.Read as DB_Read


import os
import random

import os
import random

def patterns_to_configs(config_pattern: str | tuple, random_config_order: bool) -> tuple[str, ...]:
    """
    Generate unique configurations based on the input pattern(s).
    :param config_pattern: A string or tuple of configuration patterns.
    :param random_config_order: Whether to shuffle the configurations randomly.
    :return: A tuple of unique configurations.
    """
    if not isinstance(config_pattern, tuple):
        config_pattern = (config_pattern,)
    all_configs = DB_Read.unique_configs(config_pattern)
    if random_config_order:
        random.shuffle(all_configs)
    else:
        all_configs.sort()
    return tuple(all_configs)




def save_results(config: str, epoch: int, model_stat_file: str, prm: dict):
    trials_dict_all = [prm]

    if exists(model_stat_file):
        with open(model_stat_file, 'r') as f:
            previous_trials = json.load(f)
            trials_dict_all = previous_trials + trials_dict_all

    trials_dict_all = sorted(trials_dict_all, key=lambda x: x['accuracy'], reverse=True)
    # Save trials.json
    with open(model_stat_file, 'w') as f:
        json.dump(trials_dict_all, f, indent=4)

    print(f"Trial (accuracy {prm['accuracy']}) for {config} saved at {model_stat_file}")
    DB_Write.save_results(config, epoch, model_stat_file, prm)
