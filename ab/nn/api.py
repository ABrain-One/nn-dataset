from typing import Optional

import ab.nn.util.db.Read as DB_Read
import ab.nn.util.Train as Train
from ab.nn.util.Const import default_epoch_limit_minutes
from pandas import DataFrame
import functools

from ab.nn.util.db.Query import JoinConf


@functools.lru_cache(maxsize=10)
def data(only_best_accuracy=False, task=None, dataset=None, metric=None, nn=None, epoch=None, max_rows=None, sql: Optional[JoinConf] = None, nn_prefixes=None,
         unique_nn=False) -> DataFrame:
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
    dt: tuple[dict, ...] = DB_Read.data(only_best_accuracy, task=task, dataset=dataset, metric=metric, nn=nn, epoch=epoch, max_rows=max_rows,
                                        sql=sql, nn_prefixes=nn_prefixes, unique_nn=unique_nn)
    return DataFrame.from_records(dt)


@functools.lru_cache(maxsize=10)
def run_data(model_name=None, device_type=None, max_rows=None) -> DataFrame:
    """
    Get runtime analytics as a pandas DataFrame.

    Parameters:
      - model_name (str | None): filter by model name (FK to nn.name)
      - device_type (str | None): filter by device type
      - max_rows (int | None): maximum number of results

    Returns:
      - A pandas DataFrame with columns:
        'id', 'model_name', 'device_type', 'os_version', 'valid', 'emulator', 'error_message', 'duration', 'device_analytics'
    """
    dt: tuple[dict, ...] = DB_Read.run_data(model_name=model_name, device_type=device_type, max_rows=max_rows)
    return DataFrame.from_records(dt)


def check_nn(nn_code: str, task: str, dataset: str, metric: str, prm: dict, save_to_db=True, prefix=None, save_path=None, export_onnx=False,
             epoch_limit_minutes=default_epoch_limit_minutes, transform_dir=None) -> tuple[str, float, float, float]:
    """
    Train the new NN model with the provided hyperparameters (prm) and save it to the database if training is successful.
    for argument description see :ref:`ab.nn.util.db.Write.save_nn()`
    :return: Automatically generated name of NN model, its accuracy, accuracy to time metric, and quality of the code metric.
    """
    return Train.train_new(nn_code, task, dataset, metric, prm, save_to_db=save_to_db, prefix=prefix, save_path=save_path, export_onnx=export_onnx,
                           epoch_limit_minutes=epoch_limit_minutes, transform_dir=transform_dir)
