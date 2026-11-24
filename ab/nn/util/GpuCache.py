"""GPU dataset cache: keep a whole image dataset on the GPU as uint8 and run the transform there, per batch.

For small fixed-size datasets (CIFAR-10/100, MNIST, SVHN, ...) the CPU input pipeline (PIL decode, per-sample
transforms, collate, host-to-device copy) dominates the wall-clock time of short training runs. This module
replaces torch DataLoader by GpuCachedLoader when the dataset is eligible:

  1. the base dataset holds its raw images in one array/tensor (.data) of dtype uint8 with a fixed shape,
     and its labels in .targets or .labels (torch Subset wrappers are resolved to indices);
  2. the transform is a torchvision Compose made only of GPU-expressible steps
     (Resize, CenterCrop, Pad, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, Normalize);
  3. the raw uint8 data fits a VRAM budget (default 25% of free memory); train and test splits are counted
     together, and the same raw array is uploaded only once per process (train/val share CIFAR's train array);
  4. no custom collate_fn.

Everything else keeps the standard DataLoader. Each decision is printed with the reason and the bytes.
Switch: environment variable NN_GPU_CACHE=1 (default off, so behaviour is unchanged unless asked for).
"""
import math
import os

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.v2.functional as F
from torch.utils.data import Subset

ENV_FLAG = 'NN_GPU_CACHE'
VRAM_FRACTION = float(os.environ.get('NN_GPU_CACHE_VRAM_FRACTION', '0.25'))

_SUPPORTED = (T.Resize, T.CenterCrop, T.Pad, T.RandomCrop, T.RandomHorizontalFlip, T.RandomVerticalFlip,
              T.ToTensor, T.Normalize)
_RANDOM = (T.RandomCrop, T.RandomHorizontalFlip, T.RandomVerticalFlip)

# id(raw array) -> (raw array kept alive, uint8 tensor on GPU in N,C,H,W)
_UPLOADED = {}
_PLANNED_BYTES = 0


def enabled():
    return os.environ.get(ENV_FLAG, '0').lower() in ('1', 'true', 'yes', 'on')


def _resolve(ds):
    """(base dataset, indices or None) through nested torch Subsets."""
    idx = None
    while isinstance(ds, Subset):
        sub = np.asarray(ds.indices, dtype=np.int64)
        idx = sub if idx is None else sub[idx]
        ds = ds.dataset
    return ds, idx


def _labels(base):
    for name in ('targets', 'labels'):
        y = getattr(base, name, None)
        if y is not None:
            return y
    return None


def _nchw_layout(data):
    """How to bring raw data to N,C,H,W: 'nhw' (gray), 'nhwc' or 'nchw'; None if unclear."""
    if data.ndim == 3:
        return 'nhw'
    if data.ndim == 4:
        last, second = data.shape[-1], data.shape[1]
        if last in (1, 3, 4) and second not in (1, 3, 4):
            return 'nhwc'
        if second in (1, 3, 4) and last not in (1, 3, 4):
            return 'nchw'
    return None


def _steps(transform):
    if transform is None:
        return None, 'no transform (raw PIL output)'
    if not isinstance(transform, T.Compose):
        transform = T.Compose([transform])
    for t in transform.transforms:
        if not isinstance(t, _SUPPORTED):
            return None, f'transform step {type(t).__name__} is not GPU-expressible'
        if isinstance(t, T.Pad) and t.padding_mode != 'constant':
            return None, f'Pad mode {t.padding_mode} not supported'
        if isinstance(t, T.RandomCrop) and (t.pad_if_needed or t.padding_mode != 'constant'):
            return None, 'RandomCrop with pad_if_needed/non-constant padding not supported'
    if not any(isinstance(t, T.ToTensor) for t in transform.transforms):
        return None, 'transform has no ToTensor'
    return list(transform.transforms), None


def check(ds, device=None):
    """(plan dict or None, reason string). The plan is what GpuCachedLoader needs."""
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else None
    if device is None or device.type != 'cuda':
        return None, 'no CUDA device'
    if getattr(ds, 'collate_fn', None) is not None:
        return None, 'custom collate_fn'
    base, idx = _resolve(ds)
    data = getattr(base, 'data', None)
    if data is None:
        return None, f'{type(base).__name__} has no .data array (files are loaded lazily)'
    if isinstance(data, torch.Tensor):
        dtype_ok = data.dtype == torch.uint8
    else:
        data = np.asarray(data) if isinstance(data, (list, tuple)) else data
        dtype_ok = isinstance(data, np.ndarray) and data.dtype == np.uint8
    if not dtype_ok:
        return None, f'.data dtype is not uint8 ({getattr(data, "dtype", type(data))})'
    layout = _nchw_layout(data)
    if layout is None:
        return None, f'unclear image layout {tuple(data.shape)}'
    y = _labels(base)
    if y is None or len(y) != len(data):
        return None, 'no per-sample .targets/.labels'
    steps, why = _steps(getattr(base, 'transform', None))
    if steps is None:
        return None, why
    if getattr(base, 'target_transform', None) is not None:
        return None, 'target_transform set'
    nbytes = int(np.prod(data.shape))
    already = id(data) in _UPLOADED
    free, _ = torch.cuda.mem_get_info(device)
    budget = VRAM_FRACTION * (free + sum(t.numel() for _, t in _UPLOADED.values()))
    if not already and _PLANNED_BYTES + nbytes > budget:
        return None, f'{nbytes / 2**20:.0f} MB raw does not fit budget {budget / 2**20:.0f} MB'
    return dict(base=base, data=data, idx=idx, layout=layout, labels=y, steps=steps, device=device,
                nbytes=nbytes), f'ok, {nbytes / 2**20:.1f} MB raw uint8' + (' (already on GPU)' if already else '')


def _upload(plan):
    global _PLANNED_BYTES
    data = plan['data']
    key = id(data)
    if key not in _UPLOADED:
        x = data if isinstance(data, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(data))
        if plan['layout'] == 'nhw':
            x = x.unsqueeze(1)
        elif plan['layout'] == 'nhwc':
            x = x.permute(0, 3, 1, 2)
        _UPLOADED[key] = (data, x.contiguous().to(plan['device'], non_blocking=True))
        _PLANNED_BYTES += plan['nbytes']
    return _UPLOADED[key][1]


def _pad_amounts(padding):
    """torchvision padding -> (left, top, right, bottom)."""
    if isinstance(padding, int):
        return padding, padding, padding, padding
    padding = list(padding)
    if len(padding) == 1:
        return padding * 4
    if len(padding) == 2:
        return padding[0], padding[1], padding[0], padding[1]
    return tuple(padding)


def _random_crop(x, size, generator=None):
    """Independent random crop per sample. x: B,C,H,W."""
    b, c, h, w = x.shape
    th, tw = (size, size) if isinstance(size, int) else tuple(size)
    if (h, w) == (th, tw):
        return x
    dev = x.device
    oy = torch.randint(0, h - th + 1, (b,), device=dev, generator=generator)
    ox = torch.randint(0, w - tw + 1, (b,), device=dev, generator=generator)
    rows = (oy[:, None] + torch.arange(th, device=dev))[:, :, None]  # B,th,1
    cols = (ox[:, None] + torch.arange(tw, device=dev))[:, None, :]  # B,1,tw
    bi = torch.arange(b, device=dev)[:, None, None]
    return x.permute(0, 2, 3, 1)[bi, rows, cols].permute(0, 3, 1, 2)  # B,C,th,tw


def _flip(x, p, dim, generator=None):
    m = torch.rand(x.shape[0], device=x.device, generator=generator) < p
    return torch.where(m[:, None, None, None], x.flip(dim), x)


def apply_steps(x, steps, generator=None):
    """Run the torchvision steps on a uint8 B,C,H,W batch on the GPU; returns the model input."""
    for t in steps:
        if isinstance(t, T.Resize):
            x = F.resize(x, t.size, interpolation=t.interpolation, max_size=t.max_size, antialias=True)
        elif isinstance(t, T.CenterCrop):
            x = F.center_crop(x, list(t.size))
        elif isinstance(t, T.Pad):
            x = F.pad(x, list(_pad_amounts(t.padding)), fill=t.fill)
        elif isinstance(t, T.RandomCrop):
            if t.padding is not None:
                x = F.pad(x, list(_pad_amounts(t.padding)), fill=t.fill)
            x = _random_crop(x, t.size, generator)
        elif isinstance(t, T.RandomHorizontalFlip):
            x = _flip(x, t.p, -1, generator)
        elif isinstance(t, T.RandomVerticalFlip):
            x = _flip(x, t.p, -2, generator)
        elif isinstance(t, T.ToTensor):
            x = x.float().div_(255.0) if x.dtype == torch.uint8 else x.float()
        elif isinstance(t, T.Normalize):
            mean = torch.as_tensor(t.mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            std = torch.as_tensor(t.std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            x = (x - mean) / std
    return x


class GpuCachedLoader:
    """Drop-in for DataLoader over a cached dataset: yields (inputs, labels) already on the GPU."""

    def __init__(self, ds, batch_size, shuffle, plan, generator=None):
        self.dataset = ds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = plan['steps']
        self.device = plan['device']
        self.generator = generator
        self.x = _upload(plan)
        idx = plan['idx']
        self.idx = None if idx is None else torch.as_tensor(idx, device=self.device)
        y = plan['labels']
        y = y if isinstance(y, torch.Tensor) else torch.as_tensor(np.asarray(y))
        self.y = y.to(self.device).long()
        self.n = len(self.y) if self.idx is None else len(self.idx)

    def __len__(self):
        return math.ceil(self.n / self.batch_size)

    def __iter__(self):
        order = torch.randperm(self.n, device=self.device, generator=self.generator) if self.shuffle \
            else torch.arange(self.n, device=self.device)
        if self.idx is not None:
            order = self.idx[order]
        for b in order.split(self.batch_size):
            yield apply_steps(self.x[b], self.steps, self.generator), self.y[b]


def make_loader(ds, batch_size, shuffle, fallback, name=''):
    """GpuCachedLoader if NN_GPU_CACHE is on and ds is eligible, else fallback() (the standard DataLoader)."""
    if not enabled():
        return fallback()
    plan, why = check(ds)
    tag = f'[gpu-cache]{" " + name if name else ""}'
    if plan is None:
        print(f'{tag} standard DataLoader: {why}')
        return fallback()
    print(f'{tag} {type(plan["base"]).__name__}: {why}')
    return GpuCachedLoader(ds, batch_size, shuffle, plan)
