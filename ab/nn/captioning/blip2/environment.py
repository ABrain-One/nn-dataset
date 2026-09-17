"""Fail-fast runtime contract for the tested cached BLIP-2 environment."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import sys


TESTED_PYTHON = (3, 10)
TESTED_PACKAGES = {
    "torch": "2.9.1",
    "transformers": "4.57.6",
}


class EnvironmentCompatibilityError(RuntimeError):
    """The interpreter or core ML packages differ from the tested contract."""


def validate_environment() -> None:
    current_python = sys.version_info[:2]
    if current_python != TESTED_PYTHON:
        raise EnvironmentCompatibilityError(
            "Cached BLIP-2 requires the tested Python 3.10 runtime; detected "
            f"{current_python[0]}.{current_python[1]}. Create the documented "
            "captioning virtual environment instead of continuing unpredictably."
        )
    mismatches = []
    for package, expected in TESTED_PACKAGES.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "not installed"
        # Distribution metadata excludes local CUDA suffixes such as +cu128 on
        # the tested PyTorch wheel, so compare its stable public version.
        public = installed.split("+", 1)[0]
        if public != expected:
            mismatches.append(f"{package}=={expected} required; found {installed}")
    if mismatches:
        raise EnvironmentCompatibilityError(
            "Unsupported cached BLIP-2 environment: " + "; ".join(mismatches)
        )


__all__ = [
    "EnvironmentCompatibilityError",
    "TESTED_PACKAGES",
    "TESTED_PYTHON",
    "validate_environment",
]

