import pandas as pd

from ab.nn.util.Util import *
from ab.nn.util.db.Read import code


def unique_nn(epoch_max, nns, dataset, task, metric):
    from ab.nn.api import data
    df = data(nn_prefixes=('rag-', 'unq-'), only_best_accuracy=True, task=task, dataset=dataset, metric=metric, epoch=epoch_max)
    df = pd.concat([df,
                    data(nn=nns, only_best_accuracy=True, task=task, dataset=dataset, metric=metric, epoch=epoch_max)])
    return df.sort_values(by='accuracy', ascending=False)


def unique_nn_cls(epoch_max, dataset='cifar-10', task='img-classification', metric='acc'):
    return unique_nn(epoch_max, core_nn_cls, dataset, task, metric)


def get_attr(mod, f):
    return get_obj_attr(__import__(mod, fromlist=[f]), f)


def nn_mod(*nms):
    mod = ".".join(to_nn + nms)
    code_file = ab_root_path / (mod.replace('.', '/') + '.py')
    if not code_file.exists():
        code_file.parent.mkdir(parents=True, exist_ok=True)
        mod_l = mod.split('.')
        code_file.write_text(code(mod_l[-2], mod_l[-1]))
    return mod

def get_ab_nn_attr(mod, f):
    return get_attr(nn_mod(mod), f)

def min_accuracy(dataset):
    return get_ab_nn_attr(f"loader.{dataset}", 'MINIMUM_ACCURACY')

