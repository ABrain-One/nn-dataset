import torch
from torch.utils.data import Dataset
from datasets import load_dataset

# IMDB-Wiki normalization values (ImageNet-like, since faces are natural images)
__norm_mean = (0.485, 0.456, 0.406)
__norm_dev = (0.229, 0.224, 0.225)

# For age regression, minimum accuracy is based on MAE threshold
# Using 0.01 as a baseline (any model should do better than random)
MINIMUM_ACCURACY = 0.01


def loader(transform_fn, task):
    """
    Downloads/caches the dataset from Hugging Face and returns PyTorch datasets.

    HF dataset: systemk/imdb-wiki
    Configs  : default | face | imdb | wiki
    Splits   : train | validation | test
    
    :param transform_fn: Transform function that takes normalization params (mean, std)
    :param task: Task name (e.g., 'age-regression')
    :return: Tuple of (out_shape, minimum_accuracy, train_dataset, test_dataset)
    """
    dataset_name = "systemk/imdb-wiki"
    config = "default"   # change to "face" if you prefer face-only config
    split_train = "train"
    split_test = "validation"  # fallback to "test" if validation not available

    # Apply transform with normalization parameters (consistent with other loaders)
    transform = transform_fn((__norm_mean, __norm_dev))

    raw_train = load_dataset(dataset_name, config, split=split_train)

    try:
        raw_test = load_dataset(dataset_name, config, split=split_test)
    except Exception:
        raw_test = load_dataset(dataset_name, config, split="test")

    train_ds = IMDBWIKI(raw_train, transform_fn=transform)
    test_ds = IMDBWIKI(raw_test, transform_fn=transform)

    # Output shape for age regression is (1,) - single continuous value
    out_shape = (1,)

    return out_shape, MINIMUM_ACCURACY, train_ds, test_ds


class IMDBWIKI(Dataset):
    """
    Image -> Age regression dataset

    x: transformed image tensor (C,H,W)
    y: age tensor shape (1,) float32
    """

    def __init__(self, hf_split, transform_fn=None, prefer_face_image=True):
        self.ds = hf_split
        self.transform_fn = transform_fn

        cols = set(self.ds.column_names)
        if prefer_face_image and "face_image" in cols:
            self.image_key = "face_image"
        elif "image" in cols:
            self.image_key = "image"
        else:
            raise KeyError(f"No image column found. Columns={sorted(cols)}")

        if "age" not in cols:
            raise KeyError(f"'age' column not found. Columns={sorted(cols)}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[idx]
        img = row[self.image_key]  # PIL.Image
        age = row["age"]

        if self.transform_fn is not None:
            x = self.transform_fn(img)
        else:
            # fallback: PIL -> float tensor CHW in [0,1]
            import numpy as np
            arr = np.array(img)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

        y = torch.tensor([float(age)], dtype=torch.float32)
        return x, y
