from ab.nn.api import data
from ab.nn.util.Const import core_nn_cls
import pandas as pd


def unique_nn(epoch_max, nns, dataset, task, metric):
    df = data(nn_prefixes=('rag-', 'unq-'), only_best_accuracy=True, task=task, dataset=dataset, metric=metric, epoch=epoch_max)
    df = pd.concat([df,
                    data(nn=nns, only_best_accuracy=True, task=task, dataset=dataset, metric=metric, epoch=epoch_max)])
    return df.sort_values(by='accuracy', ascending=False)


def unique_nn_cls(epoch_max, dataset='cifar-10', task='img-classification', metric='acc'):
    return unique_nn(epoch_max, core_nn_cls, dataset, task, metric)
