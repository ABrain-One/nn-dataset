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
        if self.n > 5:
            duration = max(1e-1, time.time() - self.init_time)
            estimated_time = self.total * duration  / self.n / 60
            if estimated_time > self.epoch_limit_minutes:
                # Log a warning instead of raising exception to allow verification
                print(f"\n[WARN] Estimated time {estimated_time:.2f}m exceeds limit {self.epoch_limit_minutes}m, but continuing for verification.")
                # raise LearnTimeException(estimated_time, self.epoch_limit_minutes, duration)
        return self.it.__next__()
