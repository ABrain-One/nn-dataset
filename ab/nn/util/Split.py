"""Train / validation split for loaders with a true test split.

load_dataset(...) carves a validation set out of the training split by default (test_only=True disables it); the
test split is not touched. The
validation set is the training split seen through the TEST transform (no train augmentation). The split is seeded and,
for classification, stratified, so it depends only on (number of training samples, labels, seed).

Default size: val = test size, capped at VAL_CAP_FRACTION of the training split.
"""
import copy

import numpy as np
from torch.utils.data import Subset

VAL_SEED = 42
VAL_CAP_FRACTION = 0.2

# Loaders whose "test" split is a true test split; only these get a carved-out validation split.
TRUE_TEST_LOADERS = ('cifar-10', 'cifar-100', 'mnist', 'svhn', 'celeba-gender')


def val_size(n_train, n_test, cap_frac=VAL_CAP_FRACTION):
    """Validation size: the test size, capped at cap_frac of the training split, at least 1."""
    return max(1, min(int(n_test), int(cap_frac * n_train)))


def split_indices(n_train, n_val, labels=None, seed=VAL_SEED):
    """Seeded split of range(n_train) into sorted (train_idx, val_idx) numpy arrays.

    labels given: stratified; each class gets its share of n_val (largest-remainder rounding, so the sizes add up to
    n_val exactly) and keeps its proportion in both parts. labels None: a plain seeded permutation."""
    if not 0 < n_val < n_train:
        raise ValueError(f"validation size {n_val} must be between 1 and {n_train - 1}")
    rng = np.random.default_rng(seed)
    if labels is None:
        perm = rng.permutation(n_train)
        val_idx = perm[:n_val]
    else:
        labels = np.asarray(labels)
        if len(labels) != n_train:
            raise ValueError(f"{len(labels)} labels for {n_train} training samples")
        classes, inverse = np.unique(labels, return_inverse=True)
        counts = np.bincount(inverse, minlength=len(classes))
        exact = counts * (n_val / n_train)
        quota = np.floor(exact).astype(int)
        rest = n_val - int(quota.sum())
        # largest remainders first; ties by class order (deterministic)
        order = np.lexsort((np.arange(len(classes)), -(exact - quota)))
        quota[order[:rest]] += 1
        parts = []
        for k in range(len(classes)):
            members = np.flatnonzero(inverse == k)
            parts.append(rng.permutation(members)[:quota[k]])
        val_idx = np.concatenate(parts) if parts else np.empty(0, dtype=int)
    val_mask = np.zeros(n_train, dtype=bool)
    val_mask[val_idx] = True
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def eval_view(train_set, test_set):
    """The training split seen through the test set's transform: a shallow copy of train_set whose transform (and
    torchvision's combined `transforms`) are those of test_set. Raises if the datasets have no such attributes."""
    if not hasattr(train_set, 'transform') or not hasattr(test_set, 'transform'):
        raise TypeError(f"{type(train_set).__name__} has no 'transform' attribute; pass an explicit view")
    view = copy.copy(train_set)
    view.transform = test_set.transform
    if hasattr(train_set, 'transforms') and hasattr(test_set, 'transforms'):
        view.transforms = test_set.transforms
    if hasattr(test_set, 'target_transform'):
        view.target_transform = test_set.target_transform
    return view


def dataset_labels(dataset_name, train_set):
    """Class labels of the training split for stratification, or None."""
    if dataset_name in ('cifar-10', 'cifar-100', 'mnist'):
        return np.asarray(train_set.targets)
    if dataset_name == 'svhn':
        return np.asarray(train_set.labels)
    if dataset_name == 'celeba-gender':
        # the loader's target is attribute 20, 'Male' (loader/celeba-gender.py get_gender)
        return np.asarray(train_set.attr[:, 20])
    return None


def _carry_loader_attrs(src, dst):
    for name in ('collate_fn', 'num_workers'):
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))
    return dst


class LoadedDataset(tuple):
    """load_dataset's result: the 4-tuple (out_shape, minimum_accuracy, train_set, test_set), so callers that unpack
    four values keep working, plus the attribute val_set (the validation set, or None)."""

    def __new__(cls, out_shape, minimum_accuracy, train_set, test_set, val_set=None):
        loaded = super().__new__(cls, (out_shape, minimum_accuracy, train_set, test_set))
        loaded.val_set = val_set
        return loaded


def with_validation(dataset_name, loaded, cap_frac=VAL_CAP_FRACTION, seed=VAL_SEED):
    """A loader's (out_shape, minimum_accuracy, train_set, test_set) as a LoadedDataset: for TRUE_TEST_LOADERS the
    training split is divided into train and validation (split_train_val); other datasets keep train/test (val_set
    None) with one log line."""
    out_shape, minimum_accuracy, train_set, test_set = loaded
    if dataset_name not in TRUE_TEST_LOADERS:
        print(f"[val-split] '{dataset_name}' has no true test split: train/test as before, no validation set")
        return LoadedDataset(out_shape, minimum_accuracy, train_set, test_set)
    train_part, val_part = split_train_val(train_set, test_set, labels=dataset_labels(dataset_name, train_set),
                                           cap_frac=cap_frac, seed=seed)
    return LoadedDataset(out_shape, minimum_accuracy, train_part, test_set, val_part)


def split_train_val(train_set, test_set, labels=None, cap_frac=VAL_CAP_FRACTION, seed=VAL_SEED, view=None):
    """(train_subset, val_subset): train_subset keeps the train transform, val_subset uses the test transform
    (view, default eval_view(train_set, test_set)). test_set is not changed."""
    n_train, n_test = len(train_set), len(test_set)
    train_idx, val_idx = split_indices(n_train, val_size(n_train, n_test, cap_frac), labels, seed)
    train_subset = _carry_loader_attrs(train_set, Subset(train_set, train_idx.tolist()))
    val_subset = _carry_loader_attrs(test_set, Subset(view if view is not None else eval_view(train_set, test_set),
                                                      val_idx.tolist()))
    return train_subset, val_subset
