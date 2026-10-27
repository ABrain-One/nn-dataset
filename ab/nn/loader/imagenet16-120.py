"""NAS-Bench-201 ImageNet16-120 loader.

Place the official train_data_batch_1 ... train_data_batch_10 and
val_data files in data/ImageNet16. No automatic download is attempted.

The full official validation partition is returned, not the benchmark's
separate x-valid/x-test subsets. Transforms are supplied by nn-dataset.
"""
import hashlib
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ab.nn.util.Const import data_dir

__norm_mean = tuple(x / 255 for x in (122.68, 116.66, 104.01))
__norm_dev = tuple(x / 255 for x in (63.22, 61.26, 65.09))
CLASS_QUANTITY = 120
MINIMUM_ACCURACY = 1.0 / CLASS_QUANTITY


class ImageNet16(Dataset):
    train_list = (
        ('train_data_batch_1', '27846dcaa50de8e21a7d1a35f30f0e91'),
        ('train_data_batch_2', 'c7254a054e0e795c69120a5727050e3f'),
        ('train_data_batch_3', '4333d3df2e5ffb114b05d2ffc19b1e87'),
        ('train_data_batch_4', '1620cdf193304f4a92677b695d70d10f'),
        ('train_data_batch_5', '348b3c2fdbb3940c4e9e834affd3b18d'),
        ('train_data_batch_6', '6e765307c242a1b3d7d5ef9139b48945'),
        ('train_data_batch_7', '564926d8cbf8fc4818ba23d2faac7564'),
        ('train_data_batch_8', 'f4755871f718ccb653440b9dd0ebac66'),
        ('train_data_batch_9', 'bb6dd660c38c58552125b1a92f86b5d4'),
        ('train_data_batch_10', '8f03f34ac4b42271a294f91bf480f29b'),
    )
    valid_list = (('val_data', '3410e3017fdaefba8d5073aaa65e4bd6'),)

    def __init__(self, root, train, transform=None):
        self.root = Path(root)
        self.transform = transform
        files = self.train_list if train else self.valid_list
        # Validate the entire requested partition before unpickling anything.
        for name, expected in files:
            path = self.root / name
            if not path.is_file():
                raise RuntimeError(f'Missing ImageNet16 file: {path}. Install the official ImageNet16 batches; Imagenette is not compatible.')
            digest = hashlib.md5()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise RuntimeError(f'Corrupt ImageNet16 file (MD5 mismatch): {path}')

        arrays = []
        self.targets = []
        for name, _ in files:
            with (self.root / name).open('rb') as stream:
                entry = pickle.load(stream, encoding='latin1')
            labels = np.asarray(entry['labels'])
            keep = (labels >= 1) & (labels <= CLASS_QUANTITY)
            # Filter each batch before concatenating to avoid retaining all
            # 1,000 classes in memory for this 120-class dataset.
            arrays.append(entry['data'][keep].reshape(-1, 3, 16, 16).transpose(0, 2, 3, 1))
            self.targets.extend((labels[keep] - 1).tolist())
        self.data = np.concatenate(arrays, axis=0)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


def loader(transform_fn, task):
    transform = transform_fn((__norm_mean, __norm_dev))
    root = data_dir / 'ImageNet16'
    train_set = ImageNet16(root, train=True, transform=transform)
    test_set = ImageNet16(root, train=False, transform=transform)
    return (CLASS_QUANTITY,), MINIMUM_ACCURACY, train_set, test_set
