import importlib
import os
import sys
import tempfile
import time as time
from os.path import join
from typing import Union

import math
import numpy as np
import torch
from torch.cuda import OutOfMemoryError

import ab.nn.util.CodeEval as codeEvaluator
import ab.nn.util.db.Write as DB_Write
from ab.nn.util.Classes import DataRoll
from ab.nn.util.Exception import *
from ab.nn.util.Loader import load_dataset
from ab.nn.util.Util import *
from ab.nn.util.db.Calc import save_results
from ab.nn.util.db.Read import supported_transformers



def optuna_objective(trial, config, num_workers, min_lr, max_lr, min_momentum, max_momentum,
                     min_batch_binary_power, max_batch_binary_power_local, transform, fail_iterations, n_epochs, pretrained):
    # Debug: Print initial state
    print("[DEBUG] Entering optuna_objective")
    print(f"[DEBUG] config: {config}")
    print(f"[DEBUG] trial: {trial}")
    print(f"[DEBUG] num_workers: {num_workers}")
    print(f"[DEBUG] min_lr: {min_lr}, max_lr: {max_lr}")
    print(f"[DEBUG] min_momentum: {min_momentum}, max_momentum: {max_momentum}")
    print(f"[DEBUG] min_batch_binary_power: {min_batch_binary_power}, max_batch_binary_power_local: {max_batch_binary_power_local}")
    print(f"[DEBUG] transform candidates: {transform}")
    print(f"[DEBUG] fail_iterations: {fail_iterations}")
    print(f"[DEBUG] n_epochs: {n_epochs}")
    print(f"[DEBUG] pretrained: {pretrained}")

    task, dataset_name, metric, nn = config
    try:
        # Debug: Attempt to load supported hyperparameters
        print(f"[DEBUG] Loading supported hyperparameters for nn.{nn}")
        s_prm: set = get_attr(f"nn.{nn}", "supported_hyperparameters")()
        print(f"[DEBUG] Supported hyperparameters for {nn}: {s_prm}")

        # Suggest hyperparameters
        prms = {}
        for prm in s_prm:
            match prm:
                case 'lr':
                    prms[prm] = trial.suggest_float(prm, min_lr, max_lr, log=True)
                case 'momentum':
                    prms[prm] = trial.suggest_float(prm, min_momentum, max_momentum)
                case 'dropout':
                    prms[prm] = trial.suggest_float(prm, 0.0, 0.5)
                case 'pretrained':
                    prms[prm] = float(pretrained if pretrained else trial.suggest_categorical(prm, [0, 1]))
                case _:
                    prms[prm] = trial.suggest_float(prm, 0.0, 1.0)
        # Debug: Print out the suggested hyperparameters so far
        print(f"[DEBUG] Hyperparameters from s_prm: {prms}")

        # Suggest batch size
        batch_candidates = [max_batch(x) for x in range(min_batch_binary_power, max_batch_binary_power_local + 1)]
        print(f"[DEBUG] batch_candidates: {batch_candidates}")
        batch = trial.suggest_categorical('batch', batch_candidates)
        print(f"[DEBUG] Chosen batch size: {batch}")

        # Suggest transform
        transform_candidates = transform if transform else supported_transformers()
        transform_name = trial.suggest_categorical('transform', transform_candidates)
        print(f"[DEBUG] Chosen transform: {transform_name}")

        prms = merge_prm(prms, {'batch': batch, 'transform': transform_name})

        prm_str = ', '.join([f"{k}: {v}" for k, v in prms.items()])
        print(f"[DEBUG] Final hyperparameters: {prm_str}")
        print(f"[INFO] Initialize training with {prm_str}")

        # Load dataset
        print(f"[DEBUG] Loading dataset with task={task}, dataset_name={dataset_name}, transform_name={transform_name}")
        out_shape, minimum_accuracy, train_set, test_set = load_dataset(task, dataset_name, transform_name)
        print(f"[DEBUG] Dataset loaded:")
        print(f"[DEBUG]   out_shape={out_shape}")
        print(f"[DEBUG]   minimum_accuracy={minimum_accuracy}")
        print(f"[DEBUG]   train_set size={len(train_set)}")
        print(f"[DEBUG]   test_set size={len(test_set)}")

        # Check if text generation special case
        if task == 'txt-generation':
            print(f"[DEBUG] Using text generation model: {nn}")
            # Dynamically import RNN or LSTM model
            if nn.lower() == 'rnn':
                from ab.nn.nn.RNN import Net as RNNNet
                model = RNNNet(1, 256, len(train_set.chars), batch)
            elif nn.lower() == 'lstm':
                from ab.nn.nn.LSTM import Net as LSTMNet
                model = LSTMNet(1, 256, len(train_set.chars), batch, num_layers=2)
            else:
                raise ValueError(f"Unsupported text generation model: {nn}")

        print(f"[DEBUG] Creating Train instance...")
        # Create the Train object and run
        return Train(config, out_shape, minimum_accuracy, batch, f"nn.{nn}", task, train_set, test_set, metric,
                     num_workers, prms).train_n_eval(n_epochs)

    except Exception as e:
        # Debug: Print the entire exception
        print(f"[ERROR] Exception caught in optuna_objective: {e}")
        accuracy_duration = (0.0, 1)
        if isinstance(e, OutOfMemoryError):
            print("[ERROR] OutOfMemoryError encountered.")
            if max_batch_binary_power_local <= min_batch_binary_power:
                return accuracy_duration
            else:
                raise CudaOutOfMemory(batch)
        elif isinstance(e, AccuracyException):
            print(f"[WARN] AccuracyException: {e.message}")
            return e.accuracy, e.duration
        elif isinstance(e, LearnTimeException):
            print(f"[WARN] LearnTimeException: Estimated training time: {format_time(e.estimated_training_time)}, limit: {format_time(e.max_learn_seconds)}.")
            return (e.max_learn_seconds / e.estimated_training_time ) / 1e5, e.duration
        else:
            print(f"[WARN] '{nn}': failed to train. Error: {e}")
            if fail_iterations < 0:
                return accuracy_duration
            else:
                raise NNException() from e


class Train:
    def __init__(self,
                 config: tuple[str, str, str, str],
                 out_shape: tuple,
                 minimum_accuracy: float,
                 batch: int,
                 model_name,
                 task,
                 train_dataset,
                 test_dataset,
                 metric,
                 num_workers,
                 prm: dict,
                 save_to_db=True,
                 is_code=False,
                 save_path:Union[str,None]=None):
        print("[DEBUG] Train.__init__ called")
        print(f"[DEBUG] config={config}")
        print(f"[DEBUG] out_shape={out_shape}")
        print(f"[DEBUG] minimum_accuracy={minimum_accuracy}")
        print(f"[DEBUG] batch={batch}")
        print(f"[DEBUG] model_name={model_name}")
        print(f"[DEBUG] task={task}")
        print(f"[DEBUG] metric={metric}")
        print(f"[DEBUG] num_workers={num_workers}")
        print(f"[DEBUG] prm={prm}")
        print(f"[DEBUG] save_to_db={save_to_db}, is_code={is_code}, save_path={save_path}")

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

        # Debug: DataLoader creation
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch,
            shuffle=True,
            num_workers=get_obj_attr(self.train_dataset, 'num_workers', default=num_workers),
            collate_fn=lambda batch: self.train_dataset.__class__.collate_fn(batch, self.train_dataset.word2idx)
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch,
            shuffle=False,
            num_workers=get_obj_attr(self.test_dataset, 'num_workers', default=num_workers),
            collate_fn=lambda batch: self.test_dataset.__class__.collate_fn(batch, self.test_dataset.word2idx)

        )

        # Debug: Attempt to fetch first batch to determine input shape
        for input_tensor, _ in self.train_loader:
            print(f"[DEBUG] First batch input_tensor shape: {np.array(input_tensor).shape}")
            self.in_shape = np.array(input_tensor).shape
            break

        # Determine device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"[DEBUG] Using device: {device}")
        self.device = device

        # Load model
        print(f"[DEBUG] Loading model: {model_name}")
        model_net = get_attr(model_name, "Net")
        print(f"[DEBUG] model_net = {model_net}")
        print(f"[DEBUG] in_shape = {self.in_shape}")
        print(f"[DEBUG] out_shape = {out_shape}")
        print(f"[DEBUG] prm = {prm}")
        print(f"[DEBUG] device = {self.device}")
        self.model = model_net(self.in_shape, out_shape, prm, self.device)
        self.model.to(self.device)
        print("[DEBUG] Model initialized and moved to device.")

    def load_metric_function(self, metric_name):
        print(f"[DEBUG] Loading metric function for metric: {metric_name}")
        try:
            module = importlib.import_module(nn_mod('metric', metric_name))
            print(f"[DEBUG] Metric module found: {module}")
            print(self.out_shape)
            metric_fn = module.create_metric(self.out_shape)
            print(f"[DEBUG] metric_fn: {metric_fn}")
            return metric_fn
        except (ModuleNotFoundError, AttributeError) as e:
            print(f"[ERROR] Metric '{metric_name}' not found. {e}")
            raise ValueError(f"Metric '{metric_name}' not found. Ensure a corresponding file and function exist. Ensure the metric module has create_metric()") from e

    def train_n_eval(self, num_epochs):
        print(f"[DEBUG] Starting train_n_eval for {num_epochs} epochs")

        start_time = time.time_ns()
        print(f"[DEBUG] start_time (ns)={start_time}")
        print("[DEBUG] Calling model.train_setup...")
        self.model.train_setup(self.prm)

        accuracy_to_time = 0.0
        duration = sys.maxsize
        for epoch in range(1, num_epochs + 1):
            print(f"[INFO] ===== Epoch {epoch} =====")
            print("[DEBUG] Setting model to train mode.")
            self.model.train()
            print("[DEBUG] Start model.learn...")
            self.model.learn(DataRoll(self.train_loader))

            print("[DEBUG] Start eval...")
            accuracy = self.eval(self.test_loader)
            print(f"[INFO]  epoch {epoch}, accuracy={accuracy}")
            accuracy = 0.0 if math.isnan(accuracy) or math.isinf(accuracy) else accuracy
            duration = time.time_ns() - start_time
            print(f"[DEBUG] duration (ns)={duration}")

            accuracy_to_time = accuracy_to_time_metric(accuracy, self.minimum_accuracy, duration)
            print(f"[DEBUG] accuracy_to_time={accuracy_to_time}")

            if not good(accuracy, self.minimum_accuracy, duration):
                msg = (f"Accuracy is too low: {accuracy}. The minimum accepted accuracy for the '{self.config[1]}' "
                       f"dataset is {self.minimum_accuracy}.")
                print(f"[ERROR] {msg}")
                raise AccuracyException(accuracy, duration, msg)

            # We store a dictionary with updated performance stats
            prm = merge_prm(self.prm, {'duration': duration, 'accuracy': accuracy, 'uid': DB_Write.uuid4()})

            if self.save_to_db:
                print("[DEBUG] Saving results to DB / file.")
                if self.is_code:
                    # We don't want the filename to contain full codes
                    if self.save_path is None:
                        print(f"[WARN] parameter `save_Path` is None, stats will not be saved into a file.")
                    else:
                        save_path_file = join(self.save_path, f"{epoch}.json")
                        print(f"[DEBUG] saving results to {save_path_file}")
                        save_results(self.config + (epoch,), save_path_file, prm)
                else:
                    if self.save_path is None:
                        self.save_path = model_stat_dir(self.config)
                    save_path_file = join(self.save_path, f"{epoch}.json")
                    print(f"[DEBUG] saving results to {save_path_file}")
                    save_results(self.config + (epoch,), save_path_file, prm)
                    print("[DEBUG] saving results to DB_Write.save_results...")
                    DB_Write.save_results(self.config + (epoch,), prm)
        print(f"[INFO] Training finished. accuracy_to_time={accuracy_to_time}, duration={duration}")
        return accuracy_to_time, duration

    def eval(self, test_loader):
        print("[DEBUG] Setting model to eval mode.")
        self.model.eval()

        print("[DEBUG] Resetting metric function.")
        self.metric_function.reset()

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # If inputs have an extra dimension, remove it.
                if inputs.dim() == 5:
                    inputs = inputs.squeeze(1)
                # Select only the first caption per image (shape becomes [batch, seq_len])
                labels = labels[:, 0, :]
                hidden_state = self.model.rnn.init_zero_hidden(batch=inputs.size(0), device=self.device)
                outputs, _ = self.model(inputs, labels, hidden_state)
                self.metric_function(outputs, labels)
                if batch_idx % 10 == 0:
                    print(f"[DEBUG] eval batch_idx={batch_idx}, current metric state={self.metric_function.result()}")

        result = self.metric_function.result()
        print(f"[DEBUG] Final evaluation result={result}")
        return result




def train_new(nn_code, task, dataset, metric, prm, save_to_db=True, prefix:Union[str,None] = None, save_path:Union[str,None] = None):
    print("[DEBUG] train_new function called")
    print(f"[DEBUG] nn_code len={len(nn_code)} characters")
    print(f"[DEBUG] task={task}, dataset={dataset}, metric={metric}")
    print(f"[DEBUG] prm={prm}")
    print(f"[DEBUG] save_to_db={save_to_db}, prefix={prefix}, save_path={save_path}")

    if prefix is None:
        name = None
    else:
        name = prefix + "-" + DB_Write.uuid4()
        print(f"[DEBUG] name for the model: {name}")

    spec = importlib.util.find_spec("ab.nn.tmp")
    dir_path = os.path.dirname(spec.origin)

    print(f"[DEBUG] Creating a NamedTemporaryFile in dir={dir_path}")
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=True, dir=dir_path) as temp_file:
        temp_file_path = temp_file.name
        temp_filename = os.path.basename(temp_file.name).replace(".py", "")
        print(f"[DEBUG] temp_file_path={temp_file_path}")
        temp_file.write(nn_code)
        try:
            temp_file.seek(0)
            print("[DEBUG] Evaluating code with codeEvaluator.evaluate_single_file")
            res = codeEvaluator.evaluate_single_file(temp_file_path)
            print(f"[DEBUG] codeEvaluator result={res}")

            # Dynamically import the code
            print("[DEBUG] Importing the module dynamically.")
            spec = importlib.util.spec_from_file_location(f"ab.nn.tmp.{temp_filename}", temp_file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"ab.nn.tmp.{temp_filename}"] = module
            spec.loader.exec_module(module)
            print("[DEBUG] Module imported.")

            # Load dataset
            chosen_transform = prm.get('transform', None)
            print(f"[DEBUG] Loading dataset with load_dataset(task={task}, dataset={dataset}, transform={chosen_transform})")
            out_shape, minimum_accuracy, train_set, test_set = load_dataset(task, dataset, chosen_transform)
            print(f"[DEBUG] out_shape={out_shape}, minimum_accuracy={minimum_accuracy}, train_len={len(train_set)}, test_len={len(test_set)}")

            # Create the trainer
            trainer = Train(
                config=(task, dataset, metric, nn_code),
                out_shape=out_shape,
                minimum_accuracy=minimum_accuracy,
                batch=prm['batch'],
                model_name=f"tmp.{temp_filename}",
                task=task,
                train_dataset=train_set,
                test_dataset=test_set,
                metric=metric,
                num_workers=prm.get('num_workers', 1),
                prm=prm,
                save_to_db=save_to_db,
                is_code=True,
                save_path=save_path
            )
            epoch = prm['epoch']
            print(f"[DEBUG] Starting training for {epoch} epochs...")
            result, duration = trainer.train_n_eval(epoch)
            print(f"[DEBUG] train_n_eval finished, result={result}, duration={duration}")

            if save_to_db:
                # if result fits the requirement, save the model
                if good(result, minimum_accuracy, duration):
                    print(f"[DEBUG] result is good, saving model.")
                    name = DB_Write.save_nn(nn_code, task, dataset, metric, epoch, prm, force_name=name)
                    print(f"[INFO] Model saved to database with accuracy: {result}")
                else:
                    print(f"[WARN] Model accuracy {result} is below minimum threshold {minimum_accuracy}. Not saved.")
        except Exception as e:
            print(f"[ERROR] Error during training in train_new: {e}")
            raise

        return name, result, res['score'] / 100.0
