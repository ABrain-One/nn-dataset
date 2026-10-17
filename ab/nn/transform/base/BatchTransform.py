"""
batch-level augmentation applied as a SCHEDULE over the training epoch

Must match schedule_validate.py and the prompt JSONs:
  config = [[method, alpha, fraction], ...]
  Segments are applied in order: segment k is active while the epoch
  progress (current_batch / total_batches) lies inside its cumulative fraction window. 
  Fractions sum to exactly 1.0.

batch_transform(inputs, labels, current_batch, total_batches, config) -> (inputs, target_a, target_b, lam)

"""

import torch
import torch.nn.functional as F
from ab.gpt.brute.trans.augment.ScheduleValidate import validate_schedule

_VALIDATED = {}  # id(config)


def _canonical(config):
    key = id(config)
    if key not in _VALIDATED:
        canonical, report = validate_schedule(config, policy="renormalize")
        if canonical is None:
            raise ValueError(f"Invalid augment schedule: {report['reason']}")
        _VALIDATED[key] = canonical
    return _VALIDATED[key]


def batch_transform(inputs, labels, current_batch, total_batches, augment_configs):
    """Dispatch to the active segment for this batch epoch progress."""
    config = _canonical(augment_configs)
    # midpoint of the batch's window
    progress = (current_batch + 0.5) / max(total_batches, 1)

    cumulative = 0.0
    for method, alpha, frac in config:
        cumulative += frac
        if progress <= cumulative:
            return _APPLY[method](inputs, labels, alpha)
    return inputs, labels, labels, 1.0  # numeric safety net (unreachable)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _beta(alpha, device):
    return torch.distributions.Beta(alpha, alpha).sample().to(device)


def _rand_bbox(h, w, lam, device):
    """Random box covering (1 - lam) of the image area."""
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    cy = torch.randint(0, h, (1,), device=device).item()
    cx = torch.randint(0, w, (1,), device=device).item()
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    return y1, y2, x1, x2


def _batch_square_mask(b, h, w, side_frac, device):
    """[B,1,H,W] boolean mask: one random square of side side_frac per image."""
    lh, lw = max(int(h * side_frac), 1), max(int(w * side_frac), 1)
    cy = torch.randint(0, h, (b, 1, 1), device=device)
    cx = torch.randint(0, w, (b, 1, 1), device=device)
    ys = torch.arange(h, device=device).view(1, h, 1)
    xs = torch.arange(w, device=device).view(1, 1, w)
    mask = ((ys - cy).abs() <= lh // 2) & ((xs - cx).abs() <= lw // 2)
    return mask.unsqueeze(1)  # [B,1,H,W]


# --------------------------------------------------------------------------
# label-mixing methods (return real target_b and lam)
# --------------------------------------------------------------------------

def apply_cutmix(inputs, labels, alpha):
    inputs = inputs.clone()
    b, _, h, w = inputs.shape
    lam = _beta(alpha, inputs.device)
    idx = torch.randperm(b, device=inputs.device)
    y1, y2, x1, x2 = _rand_bbox(h, w, lam.item(), inputs.device)
    inputs[:, :, y1:y2, x1:x2] = inputs[idx, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (h * w))  # exact pixel ratio
    return inputs, labels, labels[idx], float(lam)


def apply_mixup(inputs, labels, alpha):
    lam = float(_beta(alpha, inputs.device))
    idx = torch.randperm(inputs.size(0), device=inputs.device)
    mixed = lam * inputs + (1.0 - lam) * inputs[idx]
    return mixed, labels, labels[idx], lam


def apply_resizemix(inputs, labels, alpha):
    """
    alpha = minimum resize scale
    """
    inputs = inputs.clone()
    b, _, h, w = inputs.shape
    idx = torch.randperm(b, device=inputs.device)
    hi = max(min(0.8, 0.999), alpha + 1e-3)
    scale = float(torch.empty(1, device=inputs.device).uniform_(alpha, hi))
    th, tw = max(int(h * scale), 2), max(int(w * scale), 2)
    src = F.interpolate(inputs[idx], size=(th, tw),
                        mode="bilinear", align_corners=False)
    y1 = int(torch.randint(0, h - th + 1, (1,)).item())
    x1 = int(torch.randint(0, w - tw + 1, (1,)).item())
    inputs[:, :, y1:y1 + th, x1:x1 + tw] = src
    lam = 1.0 - (th * tw) / (h * w)  # weight of the original image's label
    return inputs, labels, labels[idx], float(lam)


# --------------------------------------------------------------------------
# input-corruption methods (labels unchanged: target_b = labels, lam = 1)
# --------------------------------------------------------------------------

def apply_cutout(inputs, labels, alpha):
    inputs = inputs.clone()
    b, _, h, w = inputs.shape
    mask = _batch_square_mask(b, h, w, alpha, inputs.device)
    inputs.masked_fill_(mask, 0.0)
    return inputs, labels, labels, 1.0


def apply_erasing(inputs, labels, alpha):
    inputs = inputs.clone()
    b, c, h, w = inputs.shape
    mask = _batch_square_mask(b, h, w, alpha, inputs.device).expand(-1, c, -1, -1)
    noise = torch.randn_like(inputs)
    inputs[mask] = noise[mask]
    return inputs, labels, labels, 1.0


def apply_gridmask(inputs, labels, alpha):
    """alpha = kept ratio r per grid unit"""
    inputs = inputs.clone()
    b, _, h, w = inputs.shape
    d = int(torch.randint(max(h // 8, 2), max(h // 3, 3), (1,)).item())  # unit size
    hole = max(int(d * (1.0 - alpha)), 1)
    oy = int(torch.randint(0, d, (1,)).item())
    ox = int(torch.randint(0, d, (1,)).item())
    ys = (torch.arange(h, device=inputs.device) + oy) % d < hole
    xs = (torch.arange(w, device=inputs.device) + ox) % d < hole
    mask = (ys.view(1, 1, h, 1) & xs.view(1, 1, 1, w))  # holes at grid nodes
    inputs.masked_fill_(mask, 0.0)
    return inputs, labels, labels, 1.0


def apply_hide_and_seek(inputs, labels, alpha):
    """Divide each image into a 4x4 grid; drop each cell with prob alpha."""
    inputs = inputs.clone()
    b, _, h, w = inputs.shape
    g = 4
    drop = (torch.rand(b, 1, g, g, device=inputs.device) < alpha)
    mask = F.interpolate(drop.float(), size=(h, w), mode="nearest").bool()
    inputs.masked_fill_(mask, 0.0)
    return inputs, labels, labels, 1.0


def apply_none(inputs, labels, alpha):
    return inputs, labels, labels, 1.0


_APPLY = {
    "cutmix": apply_cutmix,
    "mixup": apply_mixup,
    "resizemix": apply_resizemix,
    "cutout": apply_cutout,
    "erasing": apply_erasing,
    "gridmask": apply_gridmask,
    "hide_and_seek": apply_hide_and_seek,
    "none": apply_none,
}