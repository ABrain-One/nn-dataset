#!/usr/bin/env python3

import argparse
import copy
import gc
import importlib.util
import inspect
import json
import multiprocessing
import os
import pickle
import re
import shutil
import sys
import tarfile
import time
import traceback
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DATASETS: Dict[str, Dict[str, Any]] = {
    "cifar-10": {
        "SOURCE_REPO": "NN-Dataset/checkpoints-epoch-50",
        "TARGET_REPO": "NN-Dataset/pt",
        "TARGET_PATH": "img-classification_cifar-10_acc",
        "RESULTS_FILE": "all_models_cifar-10_acc.json",
        "REVISION": None,
        "NUM_CLASSES": 10,
        "BASE_SIZE": 32,
        "MEAN": (0.4914, 0.4822, 0.4465),
        "STD": (0.2470, 0.2435, 0.2616),
        "MIN_ACCURACY": 0.2,
        "SIZES": (32, 64, 96, 128, 224, 256, 299),
    },
    "imagenet100": {
        "SOURCE_REPO": "NN-Dataset/checkpoints-imagenet100-epoch-50",
        "TARGET_REPO": "NN-Dataset/pt",
        "TARGET_PATH": "img-classification_imagenet100_acc",
        "RESULTS_FILE": "all_models_imagenet100_acc.json",
        "REVISION": None,
        "NUM_CLASSES": 100,
        "BASE_SIZE": 160,
        "MEAN": (0.485, 0.456, 0.406),
        "STD": (0.229, 0.224, 0.225),
        "MIN_ACCURACY": 0.05,
        "SIZES": (160, 224, 128, 112, 96, 64, 32, 256, 299),
    },
}

CALIB_SIZE = 256
ROUND_TO = 8

TECHNIQUES: List[Tuple[str, str, float]] = [
    ("group_norm", "structured", 0.20),
    ("taylor_global", "structured", 0.20),
    ("lamp_global", "structured", 0.20),
    ("layerwise_depth", "structured", 0.20),
    ("fpgm", "structured", 0.15),
    ("l2_local", "structured", 0.15),
    ("unstructured_global", "unstructured", 0.50),
    ("wanda_layer", "unstructured", 0.50),
    ("nm_2_4", "unstructured", 0.50),
]

script_path = Path(__file__).resolve()
dataset_root = script_path.parents[3]
if str(dataset_root) not in sys.path:
    sys.path.insert(0, str(dataset_root))

work_dir = dataset_root / "_work"
out_dir = script_path.parent
data_root = work_dir / "prune_data"
temp_dl_dir = work_dir / "temp_prune"
models_dir = dataset_root / "ab" / "nn" / "nn"
transforms_dir = dataset_root / "ab" / "nn" / "transform"
pruned_root = out_dir / "pruned_models"

for p in (data_root, temp_dl_dir):
    p.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch_pruning as tp
from huggingface_hub import (CommitOperationAdd, HfApi, create_repo, hf_hub_download,
                             list_repo_files)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
EVAL_BATCH = 250

_DS: Dict[str, Any] = {}
_DATA: Dict[str, torch.Tensor] = {}
_SRC_META: Dict[str, Any] = {}
_REPO_READY = False
_RES = 32
_USE_NORM = True
NUM_CLASSES = 10


def set_dataset(name: str):
    global NUM_CLASSES, IMAGENET100_ROOT
    IMAGENET100_ROOT = IMAGENET100_ROOT or os.environ.get("IMAGENET100_ROOT")
    _DS.clear()
    _DS.update(DATASETS[name])
    _DS["NAME"] = name
    _DS["TAG"] = _DS["TARGET_PATH"].split("/")[-1]
    NUM_CLASSES = _DS["NUM_CLASSES"]
    _DATA.clear()
    _SRC_META.clear()


def set_input(res: int, norm: bool):
    global _RES, _USE_NORM
    _RES = res
    _USE_NORM = norm




def _example() -> torch.Tensor:
    return torch.randn(1, 3, _RES, _RES)


def to_device(model: nn.Module, device: str) -> nn.Module:
    dev = torch.device(device)
    model.to(dev)
    for m in model.modules():
        for k, v in list(vars(m).items()):
            if k.startswith("_"):
                continue
            if isinstance(v, torch.device):
                setattr(m, k, dev)
            elif isinstance(v, str) and (v == "cpu" or v.startswith("cuda")):
                setattr(m, k, str(dev))
            elif isinstance(v, torch.Tensor) and k not in m._parameters and k not in m._buffers:
                setattr(m, k, v.to(dev))
    return model


def download_cifar10() -> Path:
    extract_path = data_root / "cifar-10-batches-py"
    if extract_path.exists():
        return extract_path
    os.makedirs(data_root, exist_ok=True)
    tar_path = data_root / "cifar-10-python.tar.gz"
    print("   [DATA] Downloading CIFAR-10...")
    urllib.request.urlretrieve(CIFAR_URL, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_root)
    tar_path.unlink()
    return extract_path


def _load_cifar10():
    root = download_cifar10()

    def read(name):
        with open(root / name, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"].reshape(-1, 3, 32, 32), np.array(d[b"labels"])

    xs, ys = zip(*[read(f"data_batch_{i}") for i in range(1, 6)])
    x, y = np.concatenate(xs), np.concatenate(ys)
    idx = np.random.RandomState(0).permutation(len(x))[:CALIB_SIZE]
    tx, ty = read("test_batch")
    _DATA["calib_x"], _DATA["calib_y"] = torch.from_numpy(x[idx]), torch.from_numpy(y[idx])
    _DATA["eval_x"], _DATA["eval_y"] = torch.from_numpy(tx), torch.from_numpy(ty)


IMAGENET100_ROOT: Optional[str] = None


def _lemur_imagenet100(size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    import importlib
    import torchvision.transforms as T
    mod = importlib.import_module("ab.nn.loader.imagenet100")

    def transform_fn(_norm):
        return T.Compose([T.Resize((size, size)), T.PILToTensor()])

    print("   [DATA] Loading ImageNet-100 through the LEMUR loader "
          "(it downloads from Kaggle on first use, which takes a while)")
    _shape, _min_acc, _train, test_set = mod.loader(transform_fn, "img-classification")
    xs, ys = [], []
    for img, label in test_set:
        xs.append(img if isinstance(img, torch.Tensor) else torch.as_tensor(np.asarray(img)))
        ys.append(int(label))
    return torch.stack(xs), torch.tensor(ys)


def _folder_imagenet100(size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    from PIL import Image
    root = Path(IMAGENET100_ROOT).expanduser()
    if not root.exists():
        raise RuntimeError(f"--imagenet-root path does not exist: {root}")
    val_dirs = sorted(root.glob("val.X*")) or sorted(root.glob("val*"))
    if not val_dirs:
        raise RuntimeError(f"no val folder under {root}")
    val_dir = val_dirs[0]
    classes = sorted(d.name for d in val_dir.iterdir() if d.is_dir())
    if len(classes) != NUM_CLASSES:
        raise RuntimeError(f"{val_dir} has {len(classes)} class folders, expected {NUM_CLASSES}")
    print(f"   [DATA] Reading ImageNet-100 validation images from {val_dir}")
    xs, ys = [], []
    for label, cls in enumerate(classes):
        for f in sorted((val_dir / cls).iterdir()):
            if f.suffix.lower() in (".jpeg", ".jpg", ".png"):
                img = Image.open(f).convert("RGB").resize((size, size))
                xs.append(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1))
                ys.append(label)
    if not xs:
        raise RuntimeError(f"no images found under {val_dir}")
    return torch.stack(xs), torch.tensor(ys)


def _load_imagenet100():
    size = _DS["BASE_SIZE"]
    cache = data_root / "imagenet100_images.pt"
    if cache.exists():
        d = torch.load(cache)
        x, y = d["x"], d["y"]
    else:
        x, y = _folder_imagenet100(size) if IMAGENET100_ROOT else _lemur_imagenet100(size)
        torch.save({"x": x, "y": y}, cache)
        print(f"   [DATA] Cached {len(x)} images to {cache}")
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))
    calib, held = perm[:CALIB_SIZE], perm[CALIB_SIZE:]
    _DATA["calib_x"], _DATA["calib_y"] = x[calib], y[calib]
    _DATA["eval_x"], _DATA["eval_y"] = x[held], y[held]


def _load_data():
    if _DATA:
        return
    if _DS["NAME"] == "cifar-10":
        _load_cifar10()
    else:
        _load_imagenet100()


def _prep(xb: torch.Tensor, device: str) -> torch.Tensor:
    x = xb.to(device).float().div_(255.0)
    if x.shape[-1] != _RES or x.shape[-2] != _RES:
        x = F.interpolate(x, size=(_RES, _RES), mode="bilinear", align_corners=False)
    if _USE_NORM:
        mean = torch.tensor(_DS["MEAN"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(_DS["STD"], device=device).view(1, 3, 1, 1)
        x = (x - mean) / std
    return x


def _logits(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def evaluate_accuracy(model: nn.Module, limit: Optional[int] = None) -> float:
    _load_data()
    xs, ys = _DATA["eval_x"], _DATA["eval_y"]
    if limit:
        idx = torch.randperm(len(xs), generator=torch.Generator().manual_seed(1))[:limit]
        xs, ys = xs[idx], ys[idx]
    bs = max(16, int(EVAL_BATCH * (32 / _RES) ** 2))
    to_device(model, DEVICE).eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(xs), bs):
            x = _prep(xs[i:i + bs], DEVICE)
            y = ys[i:i + bs].to(DEVICE)
            correct += (_logits(model(x)).argmax(1) == y).sum().item()
    to_device(model, "cpu")
    return correct / len(xs)


def collect_ignored(model: nn.Module) -> List[nn.Module]:
    ignored = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.out_features == NUM_CLASSES:
            ignored.append(m)
        elif isinstance(m, nn.Conv2d) and m.out_channels == NUM_CLASSES:
            ignored.append(m)
        elif "aux" in name.lower() and isinstance(m, (nn.Conv2d, nn.Linear)):
            ignored.append(m)
    return ignored


def prunable_layers(model, ignored, skip_first_conv=True):
    ids = {id(m) for m in ignored}
    layers, seen_first = [], False
    for m in model.modules():
        if not isinstance(m, (nn.Conv2d, nn.Linear)) or id(m) in ids:
            continue
        if isinstance(m, nn.Conv2d) and not seen_first:
            seen_first = True
            if skip_first_conv:
                continue
        layers.append(m)
    return layers


def make_permanent(model: nn.Module):
    for m in model.modules():
        if hasattr(m, "weight_orig"):
            prune.remove(m, "weight")


def _nm_mask(w: torch.Tensor, n: int = 2, m: int = 4) -> torch.Tensor:
    flat = w.detach().abs().reshape(w.shape[0], -1)
    usable = flat.shape[1] - flat.shape[1] % m
    mask = torch.ones_like(flat)
    if usable:
        g = flat[:, :usable].reshape(w.shape[0], -1, m)
        drop = g.topk(m - n, dim=2, largest=False).indices
        mk = torch.ones_like(g).scatter_(2, drop, 0.0)
        mask[:, :usable] = mk.reshape(w.shape[0], usable)
    return mask.reshape_as(w)


def _input_norms(model: nn.Module, layers: List[nn.Module]) -> Dict[nn.Module, torch.Tensor]:
    _load_data()
    sums: Dict[nn.Module, torch.Tensor] = {}

    def make_hook(mod):
        def hook(_, inp):
            t = inp[0].detach()
            if isinstance(mod, nn.Conv2d):
                s = t.pow(2).sum(dim=(0, 2, 3))
            else:
                s = t.reshape(-1, t.shape[-1]).pow(2).sum(dim=0)
            sums[mod] = sums[mod] + s if mod in sums else s
        return hook

    handles = [m.register_forward_pre_hook(make_hook(m)) for m in layers]
    to_device(model, DEVICE).eval()
    with torch.no_grad():
        for i in range(0, CALIB_SIZE, 64):
            model(_prep(_DATA["calib_x"][i:i + 64], DEVICE))
    for h in handles:
        h.remove()
    to_device(model, "cpu")
    return {m: s.sqrt().cpu() for m, s in sums.items()}


def _wanda_mask(m: nn.Module, norm: Optional[torch.Tensor], ratio: float) -> torch.Tensor:
    w = m.weight.detach()
    score = w.abs()
    if norm is not None:
        if isinstance(m, nn.Conv2d) and m.groups == 1 and norm.numel() == w.shape[1]:
            score = score * norm.view(1, -1, 1, 1)
        elif isinstance(m, nn.Linear) and norm.numel() == w.shape[1]:
            score = score * norm.view(1, -1)
    flat = score.reshape(w.shape[0], -1)
    k = int(flat.shape[1] * ratio)
    mask = torch.ones_like(flat)
    if k > 0:
        mask.scatter_(1, flat.topk(k, dim=1, largest=False).indices, 0.0)
    return mask.reshape_as(w)


def _apply_unstructured(model: nn.Module, tech: str, ratio: float):
    layers = prunable_layers(model, collect_ignored(model))
    if not layers:
        raise RuntimeError("no_prunable_layers")
    if tech == "unstructured_global":
        prune.global_unstructured([(m, "weight") for m in layers],
                                  pruning_method=prune.L1Unstructured, amount=ratio)
    elif tech == "nm_2_4":
        for m in layers:
            if m.weight.numel() >= 4:
                prune.custom_from_mask(m, "weight", _nm_mask(m.weight))
    elif tech == "wanda_layer":
        norms = _input_norms(model, layers)
        for m in layers:
            prune.custom_from_mask(m, "weight", _wanda_mask(m, norms.get(m), ratio))
    else:
        raise RuntimeError(f"unknown_technique:{tech}")
    for m in layers:
        w = m.weight
        if w.numel() and float((w == 0).float().mean()) > 0.95:
            raise RuntimeError("layer_collapse")


def _pruner_cls():
    for name in ("MagnitudePruner", "BasePruner", "MetaPruner"):
        cls = getattr(tp.pruner, name, None)
        if cls is not None:
            return cls
    raise RuntimeError("no_pruner_class")


def _pruner_kwargs(cls, ratio, ratio_dict, global_pruning, ignored, importance):
    sig = inspect.signature(cls.__init__).parameters
    ratio_key = "pruning_ratio" if "pruning_ratio" in sig else "ch_sparsity"
    dict_key = "pruning_ratio_dict" if "pruning_ratio_dict" in sig else "ch_sparsity_dict"
    kw = {"importance": importance, "iterative_steps": 1, "global_pruning": global_pruning,
          "ignored_layers": ignored, ratio_key: ratio}
    if ratio_dict:
        kw[dict_key] = ratio_dict
    if ROUND_TO and "round_to" in sig:
        kw["round_to"] = ROUND_TO
    return kw


def _imp(names, **kw):
    for n in names:
        cls = getattr(tp.importance, n, None)
        if cls is not None:
            try:
                return cls(**kw)
            except TypeError:
                return cls()
    raise RuntimeError(f"{names[0]}_unavailable")


def _apply_structured(model: nn.Module, tech: str, ratio: float):
    example = _example()
    ignored = collect_ignored(model)
    ratio_dict, global_pruning = None, False
    mag = ("GroupMagnitudeImportance", "MagnitudeImportance")

    if tech == "l2_local":
        importance = _imp(mag, p=2)
    elif tech == "group_norm":
        importance, global_pruning = _imp(("GroupNormImportance",) + mag, p=2, normalizer="mean"), True
    elif tech == "fpgm":
        importance = _imp(("FPGMImportance",))
    elif tech == "layerwise_depth":
        importance = _imp(mag, p=1)
        layers = [m for m in prunable_layers(model, ignored, skip_first_conv=False)
                  if not (isinstance(m, nn.Conv2d) and m.groups > 1)]
        ratio_dict = {}
        for i, m in enumerate(layers):
            pos = i / max(len(layers) - 1, 1)
            mult = 0.5 if pos < 0.33 else (1.0 if pos < 0.66 else 1.4)
            ratio_dict[m] = min(0.9, ratio * mult)
    elif tech == "taylor_global":
        importance, global_pruning = _imp(("GroupTaylorImportance", "TaylorImportance")), True
    elif tech == "lamp_global":
        importance, global_pruning = _imp(("LAMPImportance",), p=2), True
    else:
        raise RuntimeError(f"unknown_technique:{tech}")

    cls = _pruner_cls()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pruner = cls(model, example, **_pruner_kwargs(cls, ratio, ratio_dict, global_pruning,
                                                      ignored, importance))
    if any("Unwrapped parameters" in str(w.message) for w in caught):
        raise RuntimeError("unwrapped_parameters_unsupported")

    if tech == "taylor_global":
        _load_data()
        model.eval()
        model.zero_grad()
        for i in range(0, 192, 64):
            x = _prep(_DATA["calib_x"][i:i + 64], "cpu")
            y = _DATA["calib_y"][i:i + 64]
            F.cross_entropy(_logits(model(x)), y).backward()

    pruner.step()
    model.zero_grad()
    with torch.no_grad():
        out = _logits(model.eval()(example))
        if out.shape[-1] != NUM_CLASSES:
            raise RuntimeError("output_shape_changed")


def recalibrate_bn(model: nn.Module):
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    _load_data()
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    to_device(model, DEVICE).eval()
    for m in bns:
        m.train()
    with torch.no_grad():
        for i in range(0, CALIB_SIZE, 64):
            model(_prep(_DATA["calib_x"][i:i + 64], DEVICE))
    for m, mom in zip(bns, saved):
        m.momentum = mom
    to_device(model, "cpu").eval()


def build_candidate(base: nn.Module, tech: str, track: str, ratio: float) -> nn.Module:
    m = to_device(copy.deepcopy(base), "cpu").eval()
    if track == "structured":
        _apply_structured(m, tech, ratio)
    else:
        _apply_unstructured(m, tech, ratio)
    make_permanent(m)
    recalibrate_bn(m)
    return m


def load_model_code(model_name: str):
    path = models_dir / f"{model_name}.py"
    if not path.exists():
        path = models_dir / model_name
        if not path.exists() or path.is_dir():
            return None
    spec = importlib.util.spec_from_file_location(model_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[model_name] = module
    spec.loader.exec_module(module)
    return module


def instantiate_model(Net, prm=None, res: int = 32):
    cfg = {"lr": 0.001, "momentum": 0.9, "dropout": 0.0, "dropout_aux": 0.0,
           "batch": 32, "epoch": 50, "wd": 0.0, "norm_eps": 1e-5, "norm_momentum": 0.1}
    if isinstance(prm, dict):
        cfg.update(prm)
    for _ in range(12):
        try:
            return Net((1, 3, res, res), (NUM_CLASSES,), cfg, torch.device("cpu"))
        except KeyError as e:
            key = e.args[0]
            if key in cfg:
                raise
            cfg[key] = 0.0
    raise RuntimeError("could not satisfy model hyperparameters")


def resolution_from_transform(name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    f = transforms_dir / f"{name}.py"
    if f.exists():
        try:
            sizes = re.findall(r"(?:Resize|CenterCrop|RandomResizedCrop|RandomCrop)\(\s*\(?\s*(\d+)",
                               f.read_text())
            if sizes:
                return int(sizes[-1])
        except Exception:
            pass
    m = re.search(r"(\d{2,3})", name)
    if m and 16 <= int(m.group(1)) <= 512:
        return int(m.group(1))
    return None


def load_checkpoint(path: str):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def get_source_meta(revision: Optional[str]) -> Dict[str, Any]:
    if not _SRC_META:
        try:
            p = hf_hub_download(repo_id=_DS["SOURCE_REPO"], filename="all_models.json",
                                revision=revision, cache_dir=str(temp_dl_dir))
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, dict):
                _SRC_META.update(data)
        except Exception:
            pass
    return _SRC_META


def build_and_calibrate(module, sd, meta: Dict[str, Any]):
    prm = meta.get("prm") if isinstance(meta.get("prm"), dict) else None
    res0 = resolution_from_transform(meta.get("transform") or (prm or {}).get("transform"))
    candidates = ([res0] if res0 else []) + [r for r in _DS["SIZES"] if r != res0]
    if res0 and _DS["NAME"] == "imagenet100":
        candidates = [res0]
    best = None
    first_err: Optional[Exception] = None
    last_err: Optional[Exception] = None
    for res in candidates:
        try:
            set_input(res, True)
            net = instantiate_model(module.Net, prm, res)
            missing, _ = net.load_state_dict(sd, strict=False)
            if missing:
                raise RuntimeError(f"{len(missing)} missing keys, e.g. {missing[:3]}")
            to_device(net, "cpu").eval()
        except Exception as e:
            first_err = first_err or e
            last_err = e
            continue
        for norm in (True, False):
            set_input(res, norm)
            try:
                acc = evaluate_accuracy(net, 500)
            except Exception as e:
                first_err = first_err or e
                last_err = e
                continue
            if best is None or acc > best[0]:
                best = (acc, res, norm, net)
        if best and best[0] >= _DS["MIN_ACCURACY"]:
            break
    if best is None:
        raise Exception(f"no working input configuration: first error: {str(first_err)[:150]} "
                        f"| last error: {str(last_err)[:150]}")
    if best[0] < _DS["MIN_ACCURACY"]:
        raise Exception(f"accuracy {best[0]:.3f} near chance for every input configuration tried")
    set_input(best[1], best[2])
    print(f"   input size={best[1]} normalised={best[2]} probe accuracy={best[0]:.3f}")
    return best[3], best[1], best[2]


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())[:200]


def load_json_safe(path: Path, default):
    if not path.exists() or path.stat().st_size == 0:
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_family_manifests(results: List[Dict[str, Any]]):
    """Mirror the LEMUR tflite layout: one all_models.json per family/dataset folder.

    pruned_models/<structured|unstructured>/<tag>/all_models.json
    Each model is listed under the family of its best-scoring technique.
    """
    by_family: Dict[str, Dict[str, Any]] = {}
    for r in results:
        techs = [k for k in r if k != "model_name"]
        if not techs:
            continue
        best = techs[0]
        fam = r[best].get("technique_type")
        if fam not in ("structured", "unstructured"):
            continue
        by_family.setdefault(fam, {})[r["model_name"]] = {
            "technique": best,
            "technique_type": fam,
            "accuracy": r[best].get("accuracy"),
            "target_sparsity": r[best].get("target_sparsity"),
            "all_techniques": {k: r[k] for k in techs},
        }
    for fam, entries in by_family.items():
        d = pruned_root / fam / _DS["TAG"]
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / "all_models.json", dict(sorted(entries.items())))
        print(f"   manifest: {fam}/{_DS['TAG']}/all_models.json ({len(entries)} models)")


def save_pruned(model: nn.Module, folder: Path, model_name: str, tech: str, track: str,
                res: int, norm: bool, metrics: Dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{slug(model_name)}.pt"
    to_device(model, "cpu")
    payload = {"model_name": model_name, "model_state_dict": model.state_dict(),
               "technique": tech, "technique_type": track, "input_size": res,
               "normalised": norm, "metrics": metrics}
    if track == "unstructured":
        payload["masks"] = {k: (v != 0) for k, v in model.state_dict().items() if v.dim() > 1}
    try:
        if track == "structured":
            payload["full_model"] = model
        torch.save(payload, path)
    except Exception:
        payload.pop("full_model", None)
        payload["pruned_shapes"] = {k: list(v.shape) for k, v in model.state_dict().items()}
        torch.save(payload, path)
    return path


def upload_files(files: List[Tuple[str, str]], repo_id: str, token=None):
    global _REPO_READY
    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    if not _REPO_READY:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
        _REPO_READY = True
    ops = [CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local))
           for local, repo_path in files]
    while True:
        try:
            api.create_commit(repo_id=repo_id, repo_type="model", operations=ops,
                              commit_message=f"pruned models: {len(files)} files")
            print(f"    Uploaded {len(files)} files")
            return True
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "too many requests" in msg:
                print("\n   [WARN] Rate limit hit! Sleeping for 65 minutes...")
                time.sleep(65 * 60)
                continue
            print(f"    Upload failed: {e}")
            raise


def upload_local_tree(ds_name: str, args):
    """Upload already-pruned local .pt files without re-running any pruning.

    --push-hf only uploads models processed in that same run, and resume skips
    models already recorded, so finished work is never pushed. This walks
    pruned_models/<family>/<tag>/*.pt and uploads it as-is. Local copies are
    deleted after each file is confirmed present on the repo, unless
    --keep-local is given.
    """
    set_dataset(ds_name)
    token = args.hf_token or os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    repo = _DS["TARGET_REPO"]
    print("=" * 60)
    print(f"UPLOAD ONLY  {ds_name}  ->  {repo}")
    print("=" * 60)
    if not args.dry_run:
        create_repo(repo_id=repo, repo_type="model", exist_ok=True, token=token)

    for fam in ("structured", "unstructured"):
        local = pruned_root / fam / _DS["TAG"]
        path_in_repo = f"{fam}/{_DS['TARGET_PATH']}"
        pts = sorted(local.glob("*.pt")) if local.is_dir() else []
        if not pts:
            print(f"\n>> {path_in_repo}: nothing local, skipping")
            continue
        size = sum(p.stat().st_size for p in pts) / 1024 ** 3
        print(f"\n>> {path_in_repo}: {len(pts)} files, {size:.1f} GB")
        if args.dry_run:
            print("   [dry-run] not uploading")
            continue
        while True:
            try:
                api.upload_folder(folder_path=str(local), path_in_repo=path_in_repo,
                                  repo_id=repo, repo_type="model",
                                  allow_patterns=["*.pt"],
                                  commit_message=f"Pruned models: {path_in_repo}")
                print(f"   uploaded {len(pts)} files")
                break
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "too many requests" in msg:
                    print("   [WARN] rate limited, sleeping 65 minutes...")
                    time.sleep(65 * 60)
                    continue
                print(f"   [FAIL] {e}")
                return
        if args.keep_local:
            continue
        remote = {f.rsplit("/", 1)[-1]
                  for f in list_repo_files(repo, repo_type="model", token=token)
                  if f.startswith(path_in_repo + "/")}
        freed = n = 0
        kept = []
        for pt in pts:
            if pt.name in remote:
                freed += pt.stat().st_size
                pt.unlink()
                n += 1
            else:
                kept.append(pt.name)
        print(f"   deleted {n} local files, freed {freed / 1024 ** 3:.1f} GB")
        if kept:
            print(f"   KEPT {len(kept)} not confirmed on repo: {kept[:3]}")


def process_single_model(model_name: str, args) -> Dict[str, Any]:
    set_dataset(args.ds)
    torch.manual_seed(0)
    np.random.seed(0)
    m_dir = out_dir / f"tmp_run_{slug(model_name)}"
    m_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"\n>> Processing {model_name}...")
        ckpt_path = hf_hub_download(repo_id=_DS["SOURCE_REPO"], filename=f"{model_name}.pth",
                                    revision=args.revision, cache_dir=str(m_dir))
        module = load_model_code(model_name)
        if not module or not hasattr(module, "Net"):
            raise Exception("Model architecture not found")

        ckpt = load_checkpoint(ckpt_path)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

        meta = get_source_meta(args.revision).get(model_name)
        if not isinstance(meta, dict):
            meta = {}
        base, res, norm = build_and_calibrate(module, sd, meta)

        runs: Dict[str, Dict[str, Any]] = {}
        best: Optional[Dict[str, Any]] = None
        errors: List[str] = []
        for tech, track, ratio in TECHNIQUES:
            try:
                m = build_candidate(base, tech, track, ratio)
                acc = evaluate_accuracy(m)
                runs[tech] = {"technique_type": track, "target_sparsity": ratio,
                              "accuracy": round(acc, 4)}
                if best is None or acc > best["acc"]:
                    best = {"acc": acc, "model": m, "tech": tech, "track": track}
                else:
                    del m
            except Exception as e:
                errors.append(f"{tech}:{str(e)[:80]}")
            gc.collect()
        if not runs:
            raise Exception("all pruning techniques failed: " + "; ".join(errors)[:300])

        ordered = dict(sorted(runs.items(), key=lambda kv: -kv[1]["accuracy"]))
        folder = pruned_root / best["track"] / _DS["TAG"]
        path = save_pruned(best["model"], folder, model_name, best["tech"], best["track"],
                           res, norm, ordered[best["tech"]])
        print(f"   {len(runs)} techniques done, top: {best['tech']} "
              f"({best['track']}) accuracy={best['acc']:.4f}")
        return {"status": "success", "result": {"model_name": model_name, **ordered},
                "file": str(path), "track": best["track"]}
    except Exception as e:
        traceback.print_exc()
        return {"status": "failed", "reason": str(e)[:300]}
    finally:
        shutil.rmtree(m_dir, ignore_errors=True)
        gc.collect()
        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def _process_worker(model_name, args, q):
    try:
        q.put(process_single_model(model_name, args))
    except BaseException as e:
        q.put({"status": "failed", "reason": f"worker_exception: {e}"})


def list_models(args) -> Tuple[str, List[str]]:
    api = HfApi(token=args.hf_token or None)
    repo = _DS["SOURCE_REPO"]
    revision = _DS["REVISION"] or api.repo_info(repo, repo_type="model").sha
    names = sorted(f[:-4] for f in list_repo_files(repo, revision=revision) if f.endswith(".pth"))
    try:
        src_json = hf_hub_download(repo_id=repo, filename="all_models.json",
                                   revision=revision, cache_dir=str(temp_dl_dir))
        with open(src_json) as f:
            source_data = json.load(f)
        names = [n for n in names
                 if isinstance(source_data.get(n), dict)
                 and source_data[n].get("task") == "img-classification"]
    except Exception as e:
        print(f"  Warning: could not fetch all_models.json for filtering: {e}")
        excl = ("rag-", "txt-", "obj-", "seg-", "lstm", "rnn")
        names = [n for n in names if not n.lower().startswith(excl)]
    return revision, names


def run_dataset(ds_name: str, args):
    set_dataset(ds_name)
    args.ds = ds_name
    folder = pruned_root / _DS["TAG"]
    folder.mkdir(parents=True, exist_ok=True)
    results_path = folder / _DS["RESULTS_FILE"]
    skipped_path = folder / "skipped_models.json"

    print("=" * 60)
    print(f"DATASET {ds_name}: {_DS['SOURCE_REPO']} -> {_DS['TARGET_REPO']}/{_DS['TARGET_PATH']}")
    print("=" * 60)

    if ds_name == "imagenet100":
        try:
            _load_data()
            print(f"  Evaluation images ready: {len(_DATA['eval_x'])}")
        except Exception as e:
            print(f"  Cannot prepare ImageNet-100 evaluation images: {e}")
            return
        _DATA.clear()

    args.revision, names = list_models(args)
    print(f"  Pinned revision: {args.revision}")
    print(f"  {len(names)} image-classification models")

    if args.model:
        names = [args.model] if args.model in names else []
        if not names:
            print(f"  Model '{args.model}' not found in {ds_name}")
            return
    if getattr(args, "models_file", None):
        with open(args.models_file) as f:
            wanted = [ln.strip() for ln in f]
        wanted = [w for w in wanted if w and not w.startswith("#")]
        avail = set(names)
        missing = [w for w in wanted if w not in avail]
        names = [w for w in wanted if w in avail]
        print(f"  --models-file: {len(names)} of {len(wanted)} requested models "
              f"exist in {ds_name}")
        if missing:
            print(f"  not in {ds_name} (skipped): {len(missing)} -> {missing[:5]}")
        if not names:
            return
    if args.limit:
        names = names[: args.limit]

    results = load_json_safe(results_path, [])
    if isinstance(results, dict):
        results = list(results.values())
    results = [r for r in results if isinstance(r, dict) and "model_name" in r]
    skipped = load_json_safe(skipped_path, [])
    done = set()
    if not args.no_resume:
        done = {r["model_name"] for r in results}
        if not args.retry_failed:
            done |= {d.get("model") for d in skipped if isinstance(d, dict)}
        print(f"  Resume: {len(done)} models already handled will be skipped")

    ok_n = fail_n = 0
    for i, name in enumerate(names, 1):
        if name in done:
            continue
        print(f"\n{'=' * 60}\n[{i}/{len(names)}] {name}\n{'=' * 60}")
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=_process_worker, args=(name, args, q))
        p.start()
        p.join(args.timeout)
        if p.is_alive():
            print(f"   [TIMEOUT] {name} exceeded {args.timeout}s. Killing process...")
            p.terminate()
            p.join()
            out = {"status": "failed", "reason": "timeout"}
        else:
            try:
                out = q.get(timeout=5)
            except Exception:
                out = {"status": "failed", "reason": f"worker_crashed_exit_{p.exitcode}"}

        if out.get("status") == "success":
            results = [r for r in results if r["model_name"] != name] + [out["result"]]
            skipped = [d for d in skipped if d.get("model") != name]
            write_json(results_path, results)
            write_json(skipped_path, skipped)
            ok_n += 1
            if args.push_hf:
                model_file = out["file"]
                track = out.get("track", "structured")
                upload_files(
                    [(model_file,
                      f"{track}/{_DS['TARGET_PATH']}/{Path(model_file).name}")],
                    _DS["TARGET_REPO"], token=args.hf_token)
                if not args.keep_local:
                    Path(model_file).unlink(missing_ok=True)
        else:
            print(f"   [SKIP] {out.get('reason')}")
            skipped = [d for d in skipped if d.get("model") != name]
            skipped.append({"model": name, "reason": out.get("reason", "")})
            write_json(skipped_path, skipped)
            fail_n += 1
        time.sleep(1)

    write_family_manifests(results)
    print(f"\n{ds_name}: {ok_n} succeeded, {fail_n} failed")
    totals: Dict[str, List[float]] = {}
    for r in results:
        for tech, v in r.items():
            if isinstance(v, dict) and "accuracy" in v:
                totals.setdefault(tech, []).append(v["accuracy"])
    for tech, vals in sorted(totals.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"   {tech:22s} models={len(vals):4d}  mean accuracy={np.mean(vals):.4f}")
    print(f"Results: {results_path}\nSkipped: {skipped_path}")


def main():
    ap = argparse.ArgumentParser(description="Apply pruning techniques to every model")
    ap.add_argument("--dataset", choices=["cifar-10", "imagenet100", "both"], default="both")
    ap.add_argument("--push-hf", action="store_true")
    ap.add_argument("--keep-local", action="store_true")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--models-file", type=str, default=None,
                    help="path to a text file with one model name per line; "
                         "blank lines and # comments ignored")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--imagenet-root", default=None)
    ap.add_argument("--upload-only", action="store_true",
                    help="upload existing local pruned_models/ to HF and exit; "
                         "no pruning. Deletes local copies unless --keep-local")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --upload-only: show what would be sent, send nothing")
    args = ap.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    global IMAGENET100_ROOT
    IMAGENET100_ROOT = args.imagenet_root
    if IMAGENET100_ROOT:
        os.environ["IMAGENET100_ROOT"] = IMAGENET100_ROOT

    print(f"Device: {DEVICE}")
    for ds_name in (["cifar-10", "imagenet100"] if args.dataset == "both" else [args.dataset]):
        if args.upload_only:
            upload_local_tree(ds_name, args)
        else:
            run_dataset(ds_name, args)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()