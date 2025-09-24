# File: ab/nn/util/Train.py
# Description: Universal trainer for neural network models with Optuna support and
#              integrated 'save on improvement' checkpointing.

import importlib
import sys
import math
import time as time
import numpy as np
import torch
from os.path import join
from typing import Union
from torch.cuda import OutOfMemoryError
from pathlib import Path

import ab.nn.util.db.Write as DB_Write
from ab.nn.util.Classes import DataRoll
from ab.nn.util.Exception import *
from ab.nn.util.Loader import load_dataset
from ab.nn.util.db.Calc import save_results
from ab.nn.util.db.Read import supported_transformers

# --- THIS IS THE CORRECTED IMPORT ---
from ab.nn.util.Util import (
    args, torch_device, get_attr, get_obj_attr, nn_mod, merge_prm, uuid4,
    model_stat_dir, accuracy_to_time_metric, good, max_batch, conf_to_names, order_configs,
    get_ab_nn_attr, add_categorical_if_absent
)

import optuna

debug = False


def optuna_objective(trial, config, nn_prm, num_workers, min_lr, max_lr, min_momentum, max_momentum, min_dropout,
                     max_dropout,
                     min_batch_binary_power, max_batch_binary_power_local, transform, fail_iterations, n_epochs,
                     pretrained, epoch_limit_minutes):
    task, dataset_name, metric, nn = config
    try:
        s_prm: set = get_ab_nn_attr(f"nn.{nn}", "supported_hyperparameters")()
        prms = dict(nn_prm)
        for prm in s_prm:
            if not (prm in prms and prms[prm]):
                match prm:
                    case 'lr':
                        prms[prm] = trial.suggest_float(prm, min_lr, max_lr, log=True)
                    case 'momentum':
                        prms[prm] = trial.suggest_float(prm, min_momentum, max_momentum)
                    case 'dropout':
                        prms[prm] = trial.suggest_float(prm, min_dropout, max_dropout)
                    case 'pretrained':
                        prms[prm] = float(pretrained if pretrained else trial.suggest_categorical(prm, [0, 1]))
                    case _:
                        prms[prm] = trial.suggest_float(prm, 0.0, 1.0)
        batch = add_categorical_if_absent(trial, prms, 'batch', lambda: [max_batch(x) for x in
                                                                         range(min_batch_binary_power,
                                                                               max_batch_binary_power_local + 1)])
        transform_name = add_categorical_if_absent(trial, prms, 'transform', supported_transformers, default=transform)
        prm_str = ''
        for k, v in prms.items():
            prm_str += f", {k}: {v}"
        print(f"Initialize training with {prm_str[2:]}")
        out_shape, minimum_accuracy, train_set, test_set = load_dataset(task, dataset_name, transform_name)
        return Train(config, out_shape, minimum_accuracy, batch, nn_mod('nn', nn), task, train_set, test_set, metric,
                     num_workers, prms).train_n_eval(n_epochs, epoch_limit_minutes)
    except Exception as e:
        accuracy_duration = 0.0, 0.0, 1
        if isinstance(e, OutOfMemoryError):
            if max_batch_binary_power_local <= min_batch_binary_power:
                return accuracy_duration
            else:
                raise CudaOutOfMemory(batch)
        elif isinstance(e, AccuracyException):
            print(e.message)
            return e.accuracy, accuracy_to_time_metric(e.accuracy, minimum_accuracy, e.duration), e.duration
        elif isinstance(e, LearnTimeException):
            print(
                f"Estimated training time, minutes: {format_time(e.estimated_training_time)}, but limit {format_time(e.epoch_limit_minutes)}.")
            return (e.epoch_limit_minutes / e.estimated_training_time) / 1e5, 0, e.duration
        else:
            print(f"error '{nn}': failed to train. Error: {e}")
            if fail_iterations < 0:
                return accuracy_duration
            else:
                raise NNException()


def train_loader_f(train_dataset, batch, num_workers):
    return torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True,
                                       num_workers=get_obj_attr(train_dataset, 'num_workers', default=num_workers),
                                       collate_fn=get_obj_attr(train_dataset, 'collate_fn'))


def test_loader_f(test_dataset, batch, num_workers):
    return torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False,
                                       num_workers=get_obj_attr(test_dataset, 'num_workers', default=num_workers),
                                       collate_fn=get_obj_attr(test_dataset, 'collate_fn'))


class Train:
    def __init__(self, config: tuple[str, str, str, str], out_shape: tuple, minimum_accuracy: float, batch: int,
                 nn_module, task,
                 train_dataset, test_dataset, metric, num_workers, prm: dict, save_to_db=True, is_code=False,
                 save_path: Union[str, Path] = None):
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.minimum_accuracy = minimum_accuracy
        self.out_shape = out_shape
        self.batch = batch
        self.task = task
        self.prm = prm
        self.metric_name = metric
        self.metric_function = self.load_metric_function(metric)
        self.save_to_db = save_to_db
        self.is_code = is_code
        self.save_path = save_path
        self.train_loader = train_loader_f(self.train_dataset, self.batch, num_workers)
        self.test_loader = test_loader_f(self.test_dataset, self.batch, num_workers)
        for input_tensor, _ in self.train_loader:
            self.in_shape = np.array(input_tensor).shape
            break
        self.device = torch_device()
        model_net = get_attr(nn_module, 'Net')
        self.model = model_net(self.in_shape, out_shape, prm, self.device)
        self.model.to(self.device)

    def load_metric_function(self, metric_name):
        try:
            module = importlib.import_module(nn_mod('metric', metric_name))
            return module.create_metric(self.out_shape)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(
                f"Metric '{metric_name}' not found. Ensure a corresponding file and function exist and that the metric module has create_metric().") from e

    def train_n_eval(self, num_epochs, epoch_limit_minutes):
        start_time = time.time_ns()
        self.model.train_setup(self.prm)
        accuracy_to_time = 0.0
        duration = sys.maxsize
        for epoch in range(1, num_epochs + 1):
            print(f"epoch {epoch}", flush=True)
            self.model.train()
            self.model.learn(DataRoll(self.train_loader, epoch_limit_minutes), current_epoch=epoch)
            accuracy = self.eval(self.test_loader)
            accuracy = 0.0 if math.isnan(accuracy) or math.isinf(accuracy) else accuracy
            duration = time.time_ns() - start_time
            accuracy_to_time = accuracy_to_time_metric(accuracy, self.minimum_accuracy, duration)
            if hasattr(self.model, 'save_if_best'):
                self.model.save_if_best(accuracy)
            if not good(accuracy, self.minimum_accuracy, duration):
                raise AccuracyException(accuracy, duration,
                                        f"Accuracy is too low: {accuracy}. The minimum accepted accuracy for the '{self.config[1]}' dataset is {self.minimum_accuracy}.")
            only_prm = {k: v for k, v in self.prm.items() if k not in {'uid', 'duration', 'accuracy', 'epoch'}}
            prm = merge_prm(self.prm, {'uid': uuid4(only_prm), 'duration': duration, 'accuracy': accuracy})
            if self.save_to_db:
                if self.is_code:
                    if self.save_path is None:
                        print(f"[WARN] parameter `save_path` is null, statistics will not be saved.")
                    else:
                        save_results(self.config + (epoch,), join(self.save_path, f"{epoch}.json"), prm)
                else:
                    if self.save_path is None:
                        self.save_path = model_stat_dir(self.config)
                    save_results(self.config + (epoch,), join(self.save_path, f"{epoch}.json"), prm)
                    DB_Write.save_results(self.config + (epoch,), prm)
        return accuracy, accuracy_to_time, duration

    def eval(self, test_loader):
        self.model.eval()
        self.metric_function.reset()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                if torch.is_tensor(labels):
                    labels = labels.to(self.device)
                outputs = self.model(inputs)
                self.metric_function(outputs, labels)
        return self.metric_function.result()


def main(config, nn_prm, epochs, trials, min_batch_binary_power, max_batch_binary_power,
         min_learning_rate, max_learning_rate, min_momentum, max_momentum, min_dropout, max_dropout,
         transform, nn_fail_attempts, random_config_order, workers, pretrained, epoch_limit_minutes,
         train_missing_pipelines):
    configs = [conf_to_names(c) for c in config.split(' ')]
    configs = order_configs(configs, random_config_order)

    # This feature is disabled if get_existing_pipelines is not available.
    # if train_missing_pipelines:
    #     existing_pipelines = get_existing_pipelines(configs)
    #     configs = [c for c in configs if c not in existing_pipelines]
    #     print(f"Found {len(configs)} missing pipelines to train.")

    for i, conf in enumerate(configs):
        print(f"Configuration {i + 1}/{len(configs)}: {'-'.join(conf)}")
        study = optuna.create_study(directions=["maximize", "maximize", "minimize"])
        study.optimize(lambda trial: optuna_objective(trial, conf, nn_prm, workers, min_learning_rate,
                                                      max_learning_rate, min_momentum, max_momentum, min_dropout,
                                                      max_dropout,
                                                      min_batch_binary_power, max_batch_binary_power, transform,
                                                      nn_fail_attempts, epochs, pretrained, epoch_limit_minutes),
                       n_trials=trials,
                       catch=(CudaOutOfMemory, NNException))
        print("Best trial:", study.best_trial.value)


if __name__ == '__main__':
    a = args()
    main(a.config, a.nn_prm, a.epochs, a.trials, a.min_batch_binary_power, a.max_batch_binary_power,
         a.min_learning_rate, a.max_learning_rate, a.min_momentum, a.max_momentum, a.min_dropout, a.max_dropout,
         a.transform, a.nn_fail_attempts, a.random_config_order, a.workers, a.pretrained, a.epoch_limit_minutes,
         a.train_missing_pipelines)