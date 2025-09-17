import ab.nn.util.db.Read as DB_Read
import ab.nn.util.Train as Train
import ab.nn.util.Util as Util
from ab.nn.util.Const import default_epoch_limit_minutes
from pandas import DataFrame
import functools


@functools.lru_cache(maxsize=10)
def data(only_best_accuracy=False, task=None, dataset=None, metric=None, nn=None, epoch=None, max_rows=None) -> DataFrame:
    """
    Get the NN model code and all related statistics as a pandas DataFrame.

    For the detailed description of arguments see :ref:`ab.nn.util.db.Read.data()`.
    
    Parameters:
      - only_best_accuracy (bool): If True, for each unique combination of 
          (task, dataset, metric, nn, epoch) only the row with the highest accuracy is returned.
          If False, all matching rows are returned.
      - task, dataset, metric, nn, epoch: Optional filters to restrict the results.
      - max_rows (int): Specifies the maximum number of results.

    Returns:
      - A pandas DataFrame where each row is a dictionary containing:
          'task', 'dataset', 'metric', 'metric_code',
          'nn', 'nn_code', 'epoch', 'accuracy', 'duration',
          'prm', and 'transform_code'.
    """
    dt: tuple[dict, ...] = DB_Read.data(only_best_accuracy, task=task, dataset=dataset, metric=metric, nn=nn, epoch=epoch, max_rows=max_rows)
    return DataFrame.from_records(dt)


def check_nn(nn_code: str, task: str, dataset: str, metric: str, prm: dict, save_to_db=True, prefix=None, save_path=None, export_onnx=False, epoch_limit_minutes=default_epoch_limit_minutes, transform_dir= None) -> tuple[str, float, float, float]:
    """
    Train the new NN model with the provided hyperparameters (prm) and save it to the database if training is successful.
    for argument description see :ref:`ab.nn.util.db.Write.save_nn()`
    :return: Automatically generated name of NN model, its accuracy, accuracy to time metric, and quality of the code metric.
    """
    return Train.train_new(nn_code, task, dataset, metric, prm, save_to_db=save_to_db, prefix=prefix, save_path=save_path, export_onnx=export_onnx, epoch_limit_minutes=epoch_limit_minutes, transform_dir= transform_dir)


def accuracy_to_time_metric(accuracy: float, training_duration: int, dataset: str) -> float:
    """
        Accuracy to time metric (for fixed number of training epochs) is essential for detecting the fastest accuracy improvements during neural network training.
        """
    return Util.accuracy_to_time_metric(accuracy, Util.min_accuracy(dataset), training_duration)
