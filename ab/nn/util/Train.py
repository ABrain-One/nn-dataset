import importlib
from pprint import pprint
import tempfile
import time as time
from os.path import join

import numpy as np
from torch.cuda import OutOfMemoryError

import ab.nn.util.CodeEval as codeEvaluator
import ab.nn.util.db.Write as DB_Write
from ab.nn.util.Classes import DataRoll
from ab.nn.util.Exception import *
from ab.nn.util.Loader import Loader
from ab.nn.util.Util import *
from ab.nn.util.db.Calc import save_results
from ab.nn.util.db.Read import supported_transformers


def optuna_objective(trial, config, num_workers, min_lr, max_lr, min_momentum, max_momentum,
                     min_batch_binary_power, max_batch_binary_power_local, transform, fail_iterations, n_epochs):
    task, dataset_name, metric, nn = config
    try:
        # Load model
        s_prm: set = get_attr(f"nn.{nn}", "supported_hyperparameters")()
        # Suggest hyperparameters
        prms = {}
        for prm in s_prm:
            if 'lr' == prm:
                prms[prm] = trial.suggest_float('lr', min_lr, max_lr, log=True)
            elif 'momentum' == prm:
                prms[prm] = trial.suggest_float('momentum', min_momentum, max_momentum, log=False)
            elif 'dropout' == prm:  ## Dropoout of high value will prevent the model from learning
                prms[prm] = trial.suggest_float(prm, 0.0, 0.5, log=False)
            else:
                prms[prm] = trial.suggest_float(prm, 0.0, 1.0, log=False)
        batch = trial.suggest_categorical('batch', [max_batch(x) for x in range(min_batch_binary_power, max_batch_binary_power_local + 1)])
        transform_name = trial.suggest_categorical('transform', transform if transform else supported_transformers())
        prms = merge_prm(prms, {'batch': batch, 'transform': transform_name})
        prm_str = ''
        for k, v in prms.items():
            prm_str += f", {k}: {v}"
        print(f"Initialize training with {prm_str[2:]}")
        # Load dataset
        out_shape, minimum_accuracy, train_set, test_set = Loader.load_dataset(task, dataset_name, transform_name)

        # Initialize model and trainer
        if task == 'txt-generation':
            # Dynamically import RNN or LSTM model
            if nn.lower() == 'rnn':
                from ab.nn.nn.RNN import Net as RNNNet
                model = RNNNet(1, 256, len(train_set.chars), batch)
            elif nn.lower() == 'lstm':
                from ab.nn.nn.LSTM import Net as LSTMNet
                model = LSTMNet(1, 256, len(train_set.chars), batch, num_layers=2)
            else:
                raise ValueError(f"Unsupported text generation model: {nn}")
        return Train(config, out_shape, minimum_accuracy, batch, nn, task, train_set, test_set, metric,
                     num_workers, prms).train_n_eval(n_epochs)
    except Exception as e:
        if isinstance(e, OutOfMemoryError):
            if max_batch_binary_power_local <= min_batch_binary_power:
                return 0.0
            else:
                raise CudaOutOfMemory(batch)
        elif isinstance(e, AccuracyException):
            print(e.message)
            return e.accuracy
        elif isinstance(e, LearnTimeException):
            print(f"Estimated training time: {format_time(e.estimated_training_time)}, but limit {format_time(e.max_learn_seconds)}.")
            return (1 - (e.estimated_training_time / e.max_learn_seconds)) / 10
        else:
            print(f"error '{nn}': failed to train. Error: {e}")
            if fail_iterations < 0:
                return 0.0
            else:
                raise ModelException()


class Train:
    def __init__(self, config: tuple[str, str, str, str], out_shape: tuple, minimum_accuracy: float, batch: int, model_name, task,
                 train_dataset, test_dataset, metric, num_workers, prm: dict, is_code=False, save_to_db=True):
        """
        Universal class for training CV, Text Generation and other models.
        :param config: Tuple of names (Task, Dataset, Metric, Model).
        :param out_shape: Shape of output tensor of the model (e.g., number of classes for classification tasks).
        :param batch: Batch size used for both training and evaluation.
        :param minimum_accuracy: Expected average value for accuracy provided by the untrained NN model due to random output generation. This value is essential for excluding NN models without accuracy gains.
        :param model_name: Neural network model name (e.g., 'ResNet').
        :param task: e.g., 'img-segmentation' to specify the task type.
        :param train_dataset: Dataset used for training the model (e.g., torch.utils.data.Dataset).
        :param test_dataset: Dataset used for evaluating/testing the model (e.g., torch.utils.data.Dataset).
        :param metric: Name of the evaluation metric (e.g., 'acc', 'iou').
        :param prm: Dictionary of hyperparameters and their values (e.g., {'lr': 0.11, 'momentum': 0.2})
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

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch, shuffle=True,
                                                        num_workers=get_obj_attr(self.train_dataset, 'num_workers', default=num_workers),
                                                        collate_fn=get_obj_attr(self.train_dataset, 'collate_fn'))
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch, shuffle=False,
                                                       num_workers=get_obj_attr(self.train_dataset, 'num_workers', default=num_workers),
                                                       collate_fn=get_obj_attr(self.test_dataset, 'collate_fn'))

        for input_tensor, _ in self.train_loader:
            self.in_shape = np.array(input_tensor).shape # Model input tensor shape (e.g., (8, 3, 32, 32) for a batch size 8, RGB image 32x32 px).
            break

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.device = device

        # Load model
        if is_code:
            # if model_name is model
            self.model = model_name(self.in_shape, out_shape, prm)
        else:
            # if model_name is model name
            model_net = get_attr(f"nn.{model_name}", "Net")
            self.model = model_net(self.in_shape, out_shape, prm)
        self.model.to(self.device)

    def load_metric_function(self, metric_name):
        """
        Dynamically load the metric function or class based on the metric_name.
        :param metric_name: Name of the metric (e.g., 'acc', 'iou').
        :return: Loaded metric function or initialized class.
        """
        try:
            module = importlib.import_module(nn_mod('metric', metric_name))
            if metric_name == "iou":
                return module.MIoU(self.out_shape)
            elif metric_name == "map":
                return module.MAPMetric()
            else:
                return getattr(module, "compute")
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Metric '{metric_name}' not found. Ensure a corresponding file and function exist.") \
                from e


    def train_n_eval(self, num_epochs):
        """ Training and evaluation """

        start_time = time.time_ns()
        self.model.train_setup(self.device, self.prm)
        accuracy_to_time = 0.0
        for epoch in range(1, num_epochs + 1):
            print(f"epoch {epoch}", flush=True)
            self.model.train()
            self.model.learn(DataRoll(self.train_loader))

            accuracy = self.eval(self.test_loader)
            accuracy = 0.0 if math.isnan(accuracy) or math.isinf(accuracy) else accuracy
            duration = time.time_ns() - start_time
            accuracy_to_time = accuracy_to_time_metric(accuracy, self.minimum_accuracy, duration)
            if accuracy <= self.minimum_accuracy:
                raise AccuracyException(accuracy_to_time,
                    f"Accuracy is too low: {accuracy}."
                          f" The minimum accepted accuracy for the '{self.config[1]}"
                    f"' dataset is {self.minimum_accuracy}.")
            prm = merge_prm(self.prm, {'duration': duration, 'accuracy': accuracy, 'uid': DB_Write.uuid4()})
            if self.save_to_db:
                save_results(self.config + (epoch,), join(model_stat_dir(self.config), f"{epoch}.json"), prm)
        return accuracy_to_time

    def eval(self, test_loader):
        """ Evaluation """
        self.model.eval()
        total_correct, total_samples = 0, 0
        if hasattr(self.metric_function, "reset"):  # Check for reset()
            self.metric_function.reset()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                if hasattr(self.metric_function, "update"):  # For mIoU
                    self.metric_function.update(outputs, labels)
                else:  # For accuracy and others
                    correct, total = self.metric_function(outputs, labels)
                    total_correct += correct
                    total_samples += total
        # Metric result
        if hasattr(self.metric_function, "get"):
            result = self.metric_function.get()
        else:
            result = total_correct / total_samples
        return result


def train_new(nn_code, task, dataset, metric, prm, save_to_db=True):
    """
    train the model with the given code and hyperparameters and evaluate it.

    parameter:
        nn_code (str): Code of the model
        task (str): Task type
        dataset (str): Name of the dataset
        metric (str): Evaluation metric
        prm (dict): Hyperparameters, e.g., 'lr', 'momentum', 'batch', 'epoch', 'dropout'
    return:
        (str, float): Name of the model and the accuracy
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(nn_code)  # write the code to the temp file
        res = codeEvaluator.evaluate_single_file(temp_file_path)
    try:
        pprint(f"score: {res['score']}")
        model_module = {}  # store the model class
        exec(nn_code, model_module)  # execute the code to get the model class
        ModelClass = model_module['Net']  # get the model class

        # load dataset
        out_shape, minimum_accuracy, train_set, test_set = Loader.load_dataset(task, dataset, prm.get('transform', None))

        # initialize model and trainer
        trainer = Train(
            config=(task, dataset, metric, nn_code),
            out_shape=out_shape,
            minimum_accuracy=minimum_accuracy,
            batch=prm['batch'],
            model_name=ModelClass,
            task=task,
            train_dataset=train_set,
            test_dataset=test_set,
            metric=metric,
            num_workers=prm.get('num_workers', 1),
            prm=prm,
            is_code=True,
            save_to_db=save_to_db)

        epoch = prm['epoch']
        result = trainer.train_n_eval(epoch)
        name = None
        if save_to_db:
            # if result fits the requirement, save the model to database
            if result >= minimum_accuracy:
                name = DB_Write.save_nn(nn_code, task, dataset, metric, epoch, prm)
                print(f"Model saved to database with accuracy: {result}")
            else:
                print(f"Model accuracy {result} is below the minimum threshold {minimum_accuracy}. Not saved.")

    except Exception as e:
        print(f"Error during training: {e}")
        raise

    return (name, result)
