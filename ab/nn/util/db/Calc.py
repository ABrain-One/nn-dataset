import json
import math
import re
from os.path import exists
from pathlib import Path

import ab.nn.util.db.Read as DB_Read
from ab.nn.util.Util import conf_to_names, order_configs


def sanitize_for_json(obj):
    """Recursively replace nan/inf float values with None so json.dump produces valid JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


_NAN_INF_RE = re.compile(r'\b(NaN|Infinity|-Infinity)\b')


def _safe_json_load(f):
    """Load JSON that may contain bare NaN/Infinity tokens (from earlier pipeline runs)."""
    raw = f.read()
    raw = _NAN_INF_RE.sub('null', raw)
    return json.loads(raw)


def patterns_to_configs(config_pattern: str | tuple, random_config_order: bool, train_missing_pipelines: bool) -> tuple[tuple[str, ...], ...]:
    """
    Generate unique configurations based on the input pattern(s).
    :param config_pattern: A string or tuple of configuration patterns.
    :param random_config_order: Whether to shuffle the configurations randomly.
    :param train_missing_pipelines: Find and train all missing training pipelines for provided configuration.
    :return: A tuple of unique configurations.
    """
    if not isinstance(config_pattern, (tuple, list)):
        config_pattern = [config_pattern]
    config_pattern = [conf_to_names(config) for config in config_pattern]
    configs = DB_Read.unique_configs(config_pattern)

    if train_missing_pipelines:
        tasks = set([item[0] for item in configs])
        datasets = set([item[1] for item in configs])
        metrics = set([item[2] for item in configs])
        nns = set([item[3] for item in configs])
        all_configs = []
        for task in tasks:
            for dataset in datasets:
                for metric in metrics:
                    for nn in nns:
                        all_configs.append(tuple([task, dataset, metric, nn]))
        configs = set(all_configs).difference(set([tuple(conf) for conf in configs]))

    configs = order_configs(configs, random_config_order)
    return tuple(configs)


def save_results(config_ext: tuple[str, str, str, str, int], model_stat_file: str, prm: dict):
    trials_dict_all = [prm]

    if exists(model_stat_file):
        with open(model_stat_file, 'r') as f:
            previous_trials = _safe_json_load(f)
            trials_dict_all = previous_trials + trials_dict_all

    trials_dict_all = sorted(trials_dict_all, key=lambda x: x['accuracy'], reverse=True)
    # Save trials.json
    Path(model_stat_file).parent.absolute().mkdir(parents=True, exist_ok=True)
    with open(model_stat_file, 'w') as f:
        json.dump(sanitize_for_json(trials_dict_all), f, indent=4)

    print(f"Trial (accuracy {prm['accuracy']}) for ({', '.join([str(o) for o in config_ext])}) saved at {model_stat_file}")
