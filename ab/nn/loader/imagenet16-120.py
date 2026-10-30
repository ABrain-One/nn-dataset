"""NAS-Bench-201 ImageNet16-120 loader.

Download checksum-pinned Hugging Face Parquet files automatically. Return the full 6,000-image validation partition, not the
benchmark's separate x-valid/x-test subsets.
"""
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ab.nn.util.Const import data_dir

__norm_mean = tuple(x / 255 for x in (122.68, 116.66, 104.01))
__norm_dev = tuple(x / 255 for x in (63.22, 61.26, 65.09))
CLASS_QUANTITY = 120
MINIMUM_ACCURACY = 1.0 / CLASS_QUANTITY

HF_REPO = 'ChocolateDave/imagenet-16'
HF_REVISION = '8fb32e8598c7472aff76a16776cd3504e938479f'
HF_TRAIN_FILES = (
    ('train-00000-of-00002.parquet',
     '8cdc172e1ecfb5d3a1ded2405ce50d634e5ec0f492bfbe4cffe82dc4a61ea8f7'),
    ('train-00001-of-00002.parquet',
     '330b70b60c6377440d3a3a4e73e94a4462107fc0343c6fcb5907bf46bb232273'),
)
HF_VALID_FILES = (
    ('val-00000-of-00001.parquet',
     'f9055044390728628ac629b680f680ae70dfa31ed550d73169f43e9b130220b7'),
)

def _check_hash(path, expected, algorithm):
    digest = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError(
            f'Corrupt ImageNet16 file ({algorithm} mismatch): {path}. '
            'The existing file has been preserved.'
        )

def _load_huggingface(root, train):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    files = HF_TRAIN_FILES if train else HF_VALID_FILES
    cache = root / 'huggingface'
    paths = []
    for name, expected in files:
        path = cache / 'data' / name
        if not path.is_file():
            path = Path(hf_hub_download(
                repo_id=HF_REPO,
                repo_type='dataset',
                revision=HF_REVISION,
                filename=f'data/{name}',
                local_dir=str(cache),
            ))
        _check_hash(path, expected, 'sha256')
        paths.append(path)

    # Validate every shard before decoding. Retain original row order.
    expected_count = 151700 if train else 6000
    images = np.empty((expected_count, 16, 16, 3), dtype=np.uint8)
    targets = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
                batch_size=2048, columns=['image', 'label']):
            for row in batch.to_pylist():
                label = int(row['label'])
                # Hugging Face labels are already zero-based.
                if not 0 <= label < CLASS_QUANTITY:
                    continue
                if len(targets) >= expected_count:
                    raise RuntimeError('Unexpected ImageNet16 subset size')
                with Image.open(io.BytesIO(row['image']['bytes'])) as image:
                    if image.size != (16, 16) or image.mode != 'RGB':
                        raise RuntimeError('Unexpected ImageNet16 image format')
                    images[len(targets)] = np.asarray(image)
                targets.append(label)
    if len(targets) != expected_count:
        raise RuntimeError(
            f'Unexpected ImageNet16 subset size: '
            f'{len(targets)} instead of {expected_count}'
        )
    return images, targets

class ImageNet16(Dataset):
    def __init__(self, root, train, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.data, self.targets = _load_huggingface(self.root, train)

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