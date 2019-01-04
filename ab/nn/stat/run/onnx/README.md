# ONNX FP32 runtime statistics

This directory contains ONNX FP32 runtime statistics from different execution environments.

## Layout

- `fp32/`
  - Historical ONNX FP32 statistics collected on the Windows laptop.
  - This existing directory is kept unchanged for backward compatibility and provenance.

- `fp32_barracuda_workstation/`
  - Linux workstation statistics from Experiment A using Unity Barracuda.
  - Contains six repeated completion-aware runtime campaigns: `run1` through `run6`.
  - These runs correspond exactly to the supplied source campaign folders `barracuda_completion_aware_v1` through `barracuda_completion_aware_v6`.

- `fp32_sentis_workstation/`
  - Linux workstation statistics from Experiment B using Unity Sentis / `Unity.InferenceEngine`.
  - Contains six repeated completion-aware runtime campaigns: `run1` through `run6`.
  - These runs correspond exactly to the supplied source campaign folders `inference_engine_opset12_completion_aware_v1` through `inference_engine_opset12_completion_aware_v6`.

## Naming note

`run1` through `run6` are repeated runtime campaigns, **not six software versions**. The outer directory names were normalized from the original `v1` ... `v6` packaging labels to avoid implying different Barracuda or Sentis versions.

The framework name is spelled `barracuda` throughout the new directory structure.

## Data preservation

The contents of each supplied campaign directory are copied without modifying the underlying JSON/statistics/log/manifest files; only the outer campaign directory names are changed to `run1` ... `run6`.

The pre-existing `fp32/` Windows statistics are not moved or renamed.

## Campaign contents

The Barracuda runs preserve their campaign manifests, logs, per-model ONNX FP32 statistics, and failure evidence.

The Sentis runs preserve the supplied per-model ONNX FP32 statistics, manifests, failure evidence, and the packaged convenience views (`failed_exports/`, `all_models.json`, and `skipped_models.json`) where present.

## Source-package scope

All files inside the six supplied Barracuda campaign directories and all files inside the six supplied Sentis campaign directories are included.

The Barracuda source ZIP also contained packaging-level auxiliary material outside those six final campaign directories (`failed_exports/` and a small `run/` convenience directory). Those packaging helpers are not copied into the repository because they are not one of the six final runtime campaigns and would mix auxiliary packaging state with the campaign hierarchy. No file inside `barracuda_completion_aware_v1` ... `v6` is omitted.

The Sentis source ZIP root README/index is represented by this repository README plus `WORKSTATION_CAMPAIGN_INDEX.json`; all files inside the six final Sentis campaign directories are preserved.
