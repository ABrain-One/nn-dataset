"""
A schedule is a JSON list of segments:
    [[method, alpha, fraction], ...]
where `fraction` is the fraction of the training epoch (applied in order, must sum to exactly 1.0) 
— not a probability. 
"""


METHODS = {
    "none":          (None, None, "identity; alpha must be null"),
    "cutmix":        (0.1,  2.0,  "Beta(alpha, alpha) mixing coefficient"),
    "mixup":         (0.1,  1.0,  "Beta(alpha, alpha) mixing coefficient"),
    "cutout":        (0.05, 0.5,  "hole size as fraction of image side"),
    "erasing":       (0.05, 0.5,  "erased patch size as fraction of image side; noise fill"),
    "gridmask":      (0.3,  0.7,  "kept ratio r within each grid unit (higher = less masked)"),
    "hide_and_seek": (0.05, 0.5,  "per-cell drop probability on a 4x4 grid"),
    "resizemix":     (0.1,  0.8,  "minimum resize scale of the pasted source image"),
}

SUM_TOL = 1e-3          # tolerance before renormalization kicks in
MIN_FRACTION = 1e-4     # segments below this are dropped as degenerate


class ScheduleError(ValueError):
    """Raised in 'reject' policy for invalid configs."""


def validate_schedule(config, policy="renormalize"):
    """
    Validate and canonicalize a schedule config.

    Args:
        config: list of [method, alpha, fraction]
        policy: 'renormalize' (fix what is fixable, report) or
                'reject' (raise ScheduleError on any violation)

    Returns:
        (canonical_config, report) where report is a dict:
        {valid: bool, repairs: [str, ...], reason: str|None}

    Rules:
      - non-empty list of 3-element segments
      - method in METHODS; alpha within its range (None for 'none')
      - fractions positive; renormalized to sum  1.0 (last segment absorbs rounding), 
        or rejected if policy='reject' 
      - all-'none' schedules are allowed (this is the no-augmentation baseline the benchmark needs)
    """
    repairs = []

    def fail(reason):
        if policy == "reject":
            raise ScheduleError(reason)
        return None, {"valid": False, "repairs": repairs, "reason": reason}

    if not isinstance(config, list) or len(config) == 0:
        return fail("config must be a non-empty list of segments")

    segments = []
    for i, seg in enumerate(config):
        if not isinstance(seg, (list, tuple)) or len(seg) != 3:
            return fail(f"segment {i} is not a [method, alpha, fraction] triple")
        method, alpha, frac = seg

        if method not in METHODS:
            return fail(f"segment {i}: unknown method '{method}'")

        lo, hi, _ = METHODS[method]
        if method == "none":
            if alpha is not None:
                repairs.append(f"segment {i}: alpha forced to null for 'none'")
                alpha = None
        else:
            if alpha is None:
                return fail(f"segment {i}: '{method}' requires a numeric alpha")
            alpha = float(alpha)
            if not (lo <= alpha <= hi):
                if policy == "reject":
                    raise ScheduleError(
                        f"segment {i}: alpha {alpha} outside [{lo}, {hi}] for '{method}'")
                clipped = min(max(alpha, lo), hi)
                repairs.append(f"segment {i}: alpha clipped {alpha} -> {clipped}")
                alpha = clipped
            alpha = round(alpha, 6)

        try:
            frac = float(frac)
        except (TypeError, ValueError):
            return fail(f"segment {i}: fraction '{frac}' is not numeric")
        if frac <= 0:
            repairs.append(f"segment {i}: non-positive fraction dropped")
            continue
        segments.append([method, alpha, frac])

    if not segments:
        return fail("no segments with positive fraction remain")

    total = sum(s[2] for s in segments)
    if abs(total - 1.0) > SUM_TOL:
        if policy == "reject":
            raise ScheduleError(f"fractions sum to {total:.6f}, not 1.0")
        repairs.append(f"fractions renormalized from sum {total:.6f} to 1.0")
    # Renormalize and make the sum  1.0 (last segment absorbs rounding),
    segments = [[m, a, f / total] for m, a, f in segments]
    segments = [s for s in segments if s[2] >= MIN_FRACTION] or segments
    rounded = [[m, a, round(f, 6)] for m, a, f in segments]
    rounded[-1][2] = round(1.0 - sum(s[2] for s in rounded[:-1]), 6)

    return rounded, {"valid": True, "repairs": repairs, "reason": None}