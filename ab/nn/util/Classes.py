import time as time

from tqdm import tqdm

import ab.nn.util.Const as Const
from ab.nn.util.Exception import *


class DataRoll(tqdm):
    def __init__(self, dataset, epoch_limit_minutes):
        super().__init__(dataset)
        self.it = super().__iter__()
        self.init_time = time.time()
        self.epoch_limit_minutes = epoch_limit_minutes

    def __iter__(self):
        return self

    def __next__(self):
        duration_minutes = (time.time() - self.init_time) / 60
        if duration_minutes > self.epoch_limit_minutes:
            print(f"\n[INFO] Time limit {self.epoch_limit_minutes}m reached. Finishing epoch early for evaluation.")
            raise StopIteration
        return self.it.__next__()
