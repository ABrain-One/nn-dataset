import json
import os.path

from ab.nn.util.Const import stat_train_dir
from ab.nn.util.Util import merge_prm


def gpu_type():
    return None
def cpu_type():
    return None

for p in stat_train_dir.iterdir():
    if not os.path.isdir(p):
        continue
    for epoch_file in p.iterdir():
        try:
            epoch = int(epoch_file.stem)
        except Exception:
            continue

        with open(epoch_file, 'r') as f:
            trials = json.load(f)

        trials2 = []
        for prm in trials:
            prm2 = merge_prm(prm,
                             {
                                 'gpu': gpu_type(),
                                 'cpu': cpu_type(),
                             })
            trials2.append(prm2)

        with open(epoch_file, "w") as f:
            json.dump(trials2, f, indent=4)