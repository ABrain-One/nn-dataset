import importlib, sys

import numpy as np
import time as time
import ab.nn.util.CodeEval as codeEvaluator
import ab.nn.util.db.Write as DB_Write

from os.path import join
from typing import Union
from torch.cuda import OutOfMemoryError

from ab.nn.util.Classes import DataRoll
from ab.nn.util.Exception import *
from ab.nn.util.Loader import load_dataset
from ab.nn.util.Util import *
from ab.nn.util.db.Calc import save_results
from ab.nn.util.db.Read import supported_transformers

debug = False

def optuna_objective(trial, config, nn_prm, num_workers, min_lr, max_lr, min_momentum, max_momentum, min_dropout, max_dropout,
                     min_batch_binary_power, max_batch_binary_power_local, transform, fail_iterations, n_epochs, pretrained, epoch_limit_minutes):
    task, dataset_name, metric, nn = config
    try:
        # Load model
        s_prm: set = get_ab_nn_attr(f"nn.{nn}", "supported_hyperparameters")()
        # Suggest hyperparameters
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
        batch = add_categorical_if_absent(trial, prms, 'batch', lambda: [max_batch(x) for x in range(min_batch_binary_power, max_batch_binary_power_local + 1)])
        transform_name = add_categorical_if_absent(trial, prms, 'transform', supported_transformers, default=transform)
        prm_str = ''
        for k, v in prms.items():
            prm_str += f", {k}: {v}"
        print(f"Initialize training with {prm_str[2:]}")
        # Load dataset
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
            print(f"Estimated training time, minutes: {format_time(e.estimated_training_time)}, but limit {format_time(e.epoch_limit_minutes)}.")
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
    def __init__(self, config: tuple[str, str, str, str], out_shape: tuple, minimum_accuracy: float, batch: int, nn_module, task,
                 train_dataset, test_dataset, metric, num_workers, prm: dict, save_to_db=True, is_code=False, save_path: Union[str, Path] = None):
        """
        Universal class for training CV, Text Generation and other models.
        :param config: Tuple of names (Task, Dataset, Metric, Model).
        :param out_shape: Shape of output tensor of the model (e.g., number of classes for classification tasks).
        :param batch: Batch size used for both training and evaluation.
        :param minimum_accuracy: Expected average value for accuracy provided by the untrained NN model due to random output generation. This value is essential for excluding NN models without accuracy gains.
        :param nn_module: Neural network model name (e.g., 'ab.nn.nn.ResNet', 'out.tmp.').
        :param task: e.g., 'img-segmentation' to specify the task type.
        :param train_dataset: Dataset used for training the model (e.g., torch.utils.data.Dataset).
        :param test_dataset: Dataset used for evaluating/testing the model (e.g., torch.utils.data.Dataset).
        ':param' metric: Name of the evaluation metric (e.g., 'acc', 'iou').
        :param prm: Dictionary of hyperparameters and their values (e.g., {'lr': 0.11, 'momentum': 0.2})
        :param is_code: Whether `config.model` is `nn_code` or `nn`
        :param save_path: Path to save the statistics, set to `None` to use the default
        """
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
            self.in_shape = np.array(input_tensor).shape  # Model input tensor shape (e.g., (8, 3, 32, 32) for a batch size 8, RGB image 32x32 px).
            break
        self.device = torch_device()

        # Load model
        model_net = get_attr(nn_module, 'Net')
        self.model = model_net(self.in_shape, out_shape, prm, self.device)
        self.model.to(self.device)

    def load_metric_function(self, metric_name):
        """
        Dynamically load the metric function or class based on the metric_name.
        :param metric_name: Name of the metric (e.g., 'acc', 'iou').
        :return: Loaded metric function or initialized class.
        """
        try:
            module = importlib.import_module(nn_mod('metric', metric_name))

            return module.create_metric(self.out_shape)

        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Metric '{metric_name}' not found. Ensure a corresponding file and function exist. Ensure the metric module has create_metric()") \
                from e

    def train_n_eval(self, num_epochs, epoch_limit_minutes):
        """ Training and evaluation """

        start_time = time.time_ns()
        self.model.train_setup(self.prm)
        accuracy_to_time = 0.0
        duration = sys.maxsize
        for epoch in range(1, num_epochs + 1):
            print(f"epoch {epoch}", flush=True)
            self.model.train()
            self.model.learn(DataRoll(self.train_loader, epoch_limit_minutes))

            accuracy = self.eval(self.test_loader)
            accuracy = 0.0 if math.isnan(accuracy) or math.isinf(accuracy) else accuracy
            duration = time.time_ns() - start_time
            # The accuracy-to-time metric is not stored in the database as it can change over time and can be quickly calculated from saved values.
            accuracy_to_time = accuracy_to_time_metric(accuracy, self.minimum_accuracy, duration)
            if not good(accuracy, self.minimum_accuracy, duration):
                raise AccuracyException(accuracy, duration,
                                        f"Accuracy is too low: {accuracy}."
                                        f" The minimum accepted accuracy for the '{self.config[1]}"
                                        f"' dataset is {self.minimum_accuracy}.")
            only_prm = {k: v for k, v in self.prm.items() if k not in {'uid', 'duration', 'accuracy', 'epoch'}}
            prm = merge_prm(self.prm, {'uid': uuid4(only_prm), 'duration': duration, 'accuracy': accuracy})
            if self.save_to_db:
                if self.is_code:  # We don't want the filename to contain full codes
                    if self.save_path is None:
                        print(f"[WARN]parameter `save_Path` set to null, the statics will not be saved into a file.")
                    else:
                        save_results(self.config + (epoch,), join(self.save_path, f"{epoch}.json"), prm)
                else:  # Legacy save result codes in file
                    if self.save_path is None:
                        self.save_path = model_stat_dir(self.config)
                    save_results(self.config + (epoch,), join(self.save_path, f"{epoch}.json"), prm)
                    DB_Write.save_results(self.config + (epoch,), prm)  # Separated from Calc.save_results()
        return accuracy, accuracy_to_time, duration

    def eval(self, test_loader):
        """Evaluation with standardized metric interface"""
        if debug:
            for inputs, labels in test_loader:
                print(f"[EVAL DEBUG] labels type: {type(labels)}")
                if isinstance(labels, torch.Tensor):
                    print(f"[EVAL DEBUG] labels shape: {labels.shape}")
                else:
                    print(f"[EVAL DEBUG] labels sample: {labels[:2]}")
        self.model.eval()

        # Reset the metric at the start of evaluation
        self.metric_function.reset()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                # Call the metric - all metrics now use the same interface
                self.metric_function(outputs, labels)

        # Get the final result from the metric
        return self.metric_function.result()


def train_new(nn_code, task, dataset, metric, prm, save_to_db=True, prefix: Union[str, None] = None, save_path: Union[str, None] = None, export_onnx=False, epoch_limit_minutes=default_epoch_limit_minutes, transform_dir= None):
    """
    train the model with the given code and hyperparameters and evaluate it.

    parameter:
        nn_code (str): Code of the model
        task (str): Task type
        dataset (str): Name of the dataset
        metric (str): Evaluation metric
        prm (dict): Hyperparameters, e.g., 'lr', 'momentum', 'batch', 'epoch', 'dropout'
        prefix (str|None): Prefix of the model, set to None if is unknown.
        save_path (str|None): Path to save the statistics, or None to not save.
        export_onnx (bool): Export model and its weights into ONNX file.
    return:
        (str, float): Name of the model and the accuracy
    """
    model_name = uuid4(nn_code)
    if prefix:
        model_name = prefix + "-" + model_name  # Create temporal name for processing

    tmp_modul = ".".join((out, 'nn', 'tmp'))
    tmp_modul_name = ".".join((tmp_modul, model_name))
    tmp_dir = ab_root_path / tmp_modul.replace('.', '/')
    create_file(tmp_dir, '__init__.py')
    temp_file_path = tmp_dir / f"{model_name}.py"
    trainer = None
    try:
        with open(temp_file_path, 'w') as f:
            f.write(nn_code)  # write the code to the temp file
        res = codeEvaluator.evaluate_single_file(temp_file_path)
        # load dataset
        out_shape, minimum_accuracy, train_set, test_set = load_dataset(task, dataset, prm['transform'], transform_dir)
        num_workers = prm.get('num_workers', 1)
        # initialize model and trainer
        trainer = Train(
            config=(task, dataset, metric, model_name),
            out_shape=out_shape,
            minimum_accuracy=minimum_accuracy,
            batch=prm['batch'],
            nn_module=tmp_modul_name,
            task=task,
            train_dataset=train_set,
            test_dataset=test_set,
            metric=metric,
            num_workers=num_workers,
            prm=prm,
            save_to_db=save_to_db,
            is_code=True,
            save_path=save_path)
        epoch = prm['epoch']
        accuracy, accuracy_to_time, duration = trainer.train_n_eval(epoch, epoch_limit_minutes)
        if save_to_db:
            # If the result meets the requirements, save the model to the database.
            if good(accuracy, minimum_accuracy, duration):
                model_name = DB_Write.save_nn(nn_code, task, dataset, metric, epoch, prm, force_name=model_name)
                print(f"Model saved to database with accuracy: {accuracy}")
            else:
                print(f"Model accuracy {accuracy} is below the minimum threshold {minimum_accuracy}. Not saved.")
        if export_onnx:
            for input_tensor, _ in train_loader_f(train_set, 1, num_workers):
                t = input_tensor.to(torch_device())
                export_model_to_onnx(trainer.model, t)
                break
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        remove(temp_file_path)

        try:
            del train_set
        except NameError:
            pass

        try:
            del test_set
        except NameError:
            pass

        try:
            if trainer: del trainer.model
        except NameError:
            pass
        release_memory()

    return model_name, accuracy, accuracy_to_time, res['score']
