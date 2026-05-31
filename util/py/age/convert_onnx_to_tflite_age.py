#!/usr/bin/env python3
"""
Convert an ONNX age-estimation model to TFLite and verify MAE gap.

This script is strict about the exported TFLite input shape:
- ONNX -> TFLite conversion is performed with onnx2tf using tf_converter
- the produced .tflite is checked before evaluation
- if the exported TFLite input shape is not exactly [1, H, W, 3], the script fails
- MAE evaluation no longer hides export problems by resizing the TFLite interpreter
"""

from __future__ import annotations

import argparse
import random
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
from datasets import load_dataset

# Ensure local package imports work when script is launched from scripts/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _patch_onnx_helper_for_onnx2tf() -> None:
    """Patch ONNX helper for environments where onnx2tf expects removed symbols."""
    if hasattr(onnx.helper, "float32_to_bfloat16"):
        return

    def _float32_to_bfloat16(value, truncate: bool = False):
        arr = np.asarray(value, dtype=np.float32)
        u32 = arr.view(np.uint32)
        if not truncate:
            # Round-to-nearest-even before dropping low 16 bits.
            lsb = ((u32 >> 16) & 1) + 0x7FFF
            u32 = u32 + lsb
        return (u32 >> 16).astype(np.uint16)

    onnx.helper.float32_to_bfloat16 = _float32_to_bfloat16  # type: ignore[attr-defined]


def _is_int_dim(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) > 0


def _shape_dim_to_int(value: object) -> Optional[int]:
    if _is_int_dim(value):
        return int(value)
    return None


def _infer_onnx_layout_and_size(
    onnx_path: str,
    input_height: Optional[int] = None,
    input_width: Optional[int] = None,
) -> Tuple[str, int, int, str]:
    """
    Infer ONNX layout and spatial size.

    If H/W are dynamic in the ONNX graph, input_height/input_width must be provided.
    """
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_tensor = session.get_inputs()[0]
    shape = list(input_tensor.shape)
    input_name = input_tensor.name

    if len(shape) != 4:
        raise ValueError(f"Expected ONNX 4D input, got: {shape}")

    if _shape_dim_to_int(shape[1]) == 3:
        h = _shape_dim_to_int(shape[2])
        w = _shape_dim_to_int(shape[3])
        if h is None or w is None:
            if input_height is None or input_width is None:
                h, w = 224, 224
            else:
                h, w = int(input_height), int(input_width)
        return "nchw", h, w, input_name

    if _shape_dim_to_int(shape[-1]) == 3:
        h = _shape_dim_to_int(shape[1])
        w = _shape_dim_to_int(shape[2])
        if h is None or w is None:
            if input_height is None or input_width is None:
                h, w = 224, 224
            else:
                h, w = int(input_height), int(input_width)
        return "nhwc", h, w, input_name

    raise ValueError(f"Unable to infer input layout from ONNX shape: {shape}")


def _age_bin(age: float) -> int:
    return int(age // 5)


def _stratified_split_indices(ages: List[float], train_ratio=0.70, val_ratio=0.10, seed=42):
    rng = random.Random(seed)
    bins: Dict[int, List[int]] = {}

    for idx, age in enumerate(ages):
        bins.setdefault(_age_bin(age), []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for _, idxs in bins.items():
        rng.shuffle(idxs)
        n = len(idxs)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        if n >= 3:
            n_train = min(max(1, n_train), n - 2)
            n_val = min(max(1, n_val), n - n_train - 1)
        elif n == 2:
            n_train, n_val = 1, 0
        else:
            n_train, n_val = 1, 0

        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def _prepare_samples(max_samples: int, image_h: int, image_w: int, eval_split: str, split_seed: int) -> List[Tuple[np.ndarray, float]]:
    dataset = load_dataset(
        "nu-delta/utkface",
        split="train",
        cache_dir=str(REPO_ROOT / "data" / "utkface_cache"),
    )

    norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    filtered_items = []
    for item in dataset:
        age = item.get("age")
        img = item.get("image")
        if age is None or img is None:
            continue
        try:
            age_val = float(age)
        except (TypeError, ValueError):
            continue
        if 0.0 <= age_val <= 116.0:
            filtered_items.append(item)

    ages = [float(item["age"]) for item in filtered_items]
    train_idx, val_idx, test_idx = _stratified_split_indices(ages, train_ratio=0.70, val_ratio=0.10, seed=split_seed)

    if eval_split == "train":
        chosen_indices = train_idx
    elif eval_split == "val":
        chosen_indices = val_idx
    elif eval_split in ("heldout", "test"):
        chosen_indices = test_idx
    else:
        raise ValueError(f"Unsupported eval_split: {eval_split}")

    if max_samples > 0:
        chosen_indices = chosen_indices[:max_samples]

    samples: List[Tuple[np.ndarray, float]] = []
    for idx in chosen_indices:
        item = filtered_items[idx]
        if len(samples) >= max_samples > 0:
            break

        img = item["image"].convert("RGB").resize((image_w, image_h))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - norm_mean) / norm_std

        # Keep canonical in-memory format as CHW; model-specific conversion happens later.
        chw = np.transpose(arr, (2, 0, 1)).astype(np.float32)
        age = float(item["age"])
        samples.append((chw, age))

    return samples


def _extract_training_reference_mae(summary_path: str) -> Tuple[float | None, float | None]:
    if not summary_path:
        return None, None
    p = Path(summary_path)
    if not p.exists():
        return None, None

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ts = data.get("training_summary", {})
    best_val_mae = ts.get("best_val_mae_years")
    final_val_mae = ts.get("final_val_mae_years")
    return best_val_mae, final_val_mae


def _prepare_input_for_model(sample_chw: np.ndarray, input_shape: List[int]) -> Tuple[np.ndarray, int]:
    """
    Convert a CHW sample to model input shape and pad batch if static batch > 1.

    Returns:
      input_tensor, valid_batch_items
    """
    if len(input_shape) != 4:
        raise ValueError(f"Expected 4D input, got shape spec: {input_shape}")

    shape = list(input_shape)
    batch_dim = shape[0]

    # Handle NCHW vs NHWC
    if shape[1] == 3:
        one = np.expand_dims(sample_chw, axis=0)
    elif shape[-1] == 3:
        one = np.expand_dims(np.transpose(sample_chw, (1, 2, 0)), axis=0)
    else:
        raise ValueError(f"Cannot detect channel layout from input shape: {shape}")

    if isinstance(batch_dim, str):
        return one, 1

    if batch_dim in (-1, 0, 1):
        return one, 1

    if batch_dim > 1:
        padded = np.repeat(one, repeats=batch_dim, axis=0)
        return padded, 1

    raise ValueError(f"Unsupported batch dim in input shape: {shape}")


def _prepare_input_for_tflite_model(
    sample_chw: np.ndarray,
    expected_h: int,
    expected_w: int,
    dtype: np.dtype,
) -> np.ndarray:
    """
    TFLite input is expected to be NHWC [1, H, W, 3] for this export path.
    """
    if sample_chw.shape != (3, expected_h, expected_w):
        raise ValueError(
            f"Sample shape mismatch. Got {sample_chw.shape}, expected (3, {expected_h}, {expected_w})."
        )

    nhwc = np.transpose(sample_chw, (1, 2, 0))[None, ...]
    return nhwc.astype(dtype)


def _evaluate_onnx_mae(onnx_path: str, samples: List[Tuple[np.ndarray, float]]) -> float:
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]

    total_abs = 0.0
    total = 0
    for sample, gt in samples:
        x_in, valid = _prepare_input_for_model(sample, inp.shape)
        pred = session.run([out.name], {inp.name: x_in})[0]
        pred_age = float(np.asarray(pred).reshape(-1)[0])
        total_abs += abs(pred_age - gt)
        total += valid

    return total_abs / total if total else 0.0


def _read_tflite_input_shape(tflite_path: str) -> Tuple[List[int], List[int], np.dtype]:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    input_details = interpreter.get_input_details()[0]

    raw_shape = [int(x) for x in input_details["shape"]]
    sig = input_details.get("shape_signature", input_details["shape"])
    sig_shape = [int(x) for x in sig]
    dtype = input_details["dtype"]

    return raw_shape, sig_shape, dtype


def _verify_exported_tflite_shape(
    tflite_path: str,
    expected_h: int,
    expected_w: int,
) -> Tuple[List[int], List[int], np.dtype]:
    raw_shape, sig_shape, dtype = _read_tflite_input_shape(tflite_path)
    expected_shape = [1, expected_h, expected_w, 3]

    print(f"TFLite exported input shape     : {raw_shape}")
    print(f"TFLite exported shape signature : {sig_shape}")
    print(f"Expected exported input shape   : {expected_shape}")

    if raw_shape != expected_shape:
        raise RuntimeError(
            "Exported TFLite input shape is wrong for latency use.\n"
            f"Got raw shape     : {raw_shape}\n"
            f"Got shape sig     : {sig_shape}\n"
            f"Expected raw shape: {expected_shape}\n"
            "Refusing to continue because runtime resizing would mask a bad export."
        )

    return raw_shape, sig_shape, dtype


def _convert_onnx_to_tflite(
    onnx_path: str,
    tflite_path: str,
    workdir: str,
) -> Tuple[str, str, int, int, str]:
    _patch_onnx_helper_for_onnx2tf()
    from onnx2tf import convert as onnx2tf_convert

    layout, h, w, input_name = _infer_onnx_layout_and_size(onnx_path)
    saved_model_dir = Path(workdir) / "saved_model"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    if layout == "nchw":
        overwrite_shape = [f"{input_name}:1,3,{h},{w}"]
    else:
        overwrite_shape = [f"{input_name}:1,{h},{w},3"]
    shape_hints = [overwrite_shape[0]]

    print(f"ONNX input name               : {input_name}")
    print(f"ONNX inferred layout          : {layout}")
    print(f"ONNX spatial size             : {h}x{w}")
    print(f"onnx2tf overwrite_input_shape : {overwrite_shape[0]}")

    onnx2tf_convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=str(saved_model_dir),
        overwrite_input_shape=overwrite_shape,
        shape_hints=shape_hints,
        keep_shape_absolutely_input_names=[input_name],
        batch_size=1,
        output_signaturedefs=True,
        not_use_onnxsim=False,
        non_verbose=True,
        verbosity="error",
    )

    dst = Path(tflite_path)
    os.makedirs(dst.parent, exist_ok=True)
    if dst.exists():
        dst.unlink()

    loaded_model = tf.saved_model.load(str(saved_model_dir))
    serving_fn = loaded_model.signatures["serving_default"]
    input_keys = list(serving_fn.structured_input_signature[1].keys())
    output_keys = list(serving_fn.structured_outputs.keys())
    if len(output_keys) != 1:
        raise RuntimeError(f"Expected one output tensor from SavedModel, got: {output_keys}")
    output_key = output_keys[0]

    @tf.function(input_signature=[tf.TensorSpec([1, h, w, 3], tf.float32, name="input")])
    def wrapped_serve(x: tf.Tensor) -> tf.Tensor:
        x_nchw = tf.transpose(x, [0, 3, 1, 2])
        if input_keys:
            result = serving_fn(**{input_keys[0]: x_nchw})
        else:
            result = serving_fn(x_nchw)
        return result[output_key]

    concrete_fn = wrapped_serve.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    converter.allow_custom_ops = False
    tflite_model = converter.convert()
    dst.write_bytes(tflite_model)

    return str(saved_model_dir), str(dst), h, w, layout


def _evaluate_tflite_mae(
    tflite_path: str,
    samples: List[Tuple[np.ndarray, float]],
    expected_h: int,
    expected_w: int,
) -> float:
    """
    Evaluate TFLite MAE without resizing the interpreter.

    This intentionally fails if the model was exported with the wrong input shape,
    because the goal is to validate the file that will later be used for latency tests.
    """
    raw_shape, sig_shape, input_dtype = _verify_exported_tflite_shape(
        tflite_path,
        expected_h,
        expected_w,
    )

    expected_shape = [1, expected_h, expected_w, 3]
    if raw_shape != expected_shape:
        raise RuntimeError(
            f"TFLite input shape mismatch before evaluation: raw={raw_shape}, sig={sig_shape}"
        )

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    in_idx = inp["index"]
    out_idx = out["index"]

    # float32 TFLite expected in this pipeline; still keep robust casting.
    input_dtype = inp["dtype"]

    total_abs = 0.0
    total = 0
    for sample, gt in samples:
        x_feed = _prepare_input_for_tflite_model(
            sample_chw=sample,
            expected_h=expected_h,
            expected_w=expected_w,
            dtype=input_dtype,
        )

        interpreter.set_tensor(in_idx, x_feed)
        interpreter.invoke()
        pred = interpreter.get_tensor(out_idx)
        pred_age = float(np.asarray(pred).reshape(-1)[0])

        total_abs += abs(pred_age - gt)
        total += 1

    return total_abs / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ONNX age model to TFLite and compare MAE")
    parser.add_argument(
        "--onnx-path",
        default="cluster_results/age-estimation-visualization/models/MobileAgeNet/best_model.onnx",
        help="Path to input ONNX model",
    )
    parser.add_argument(
        "--tflite-path",
        default="cluster_results/age-estimation-visualization/models/MobileAgeNet/best_model.tflite",
        help="Path to output TFLite model",
    )
    parser.add_argument(
        "--report-path",
        default="ab/nn/imp/TFLITE_REPORT_AGE_EPOCH50.json",
        help="JSON report output path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Number of samples to evaluate (0 means full selected split)",
    )
    parser.add_argument(
        "--eval-split",
        default="val",
        choices=["train", "val", "heldout", "test"],
        help="UTKFace split to evaluate, matching training loader split strategy",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for the age-stratified split (must match training for consistency)",
    )
    parser.add_argument(
        "--mae-gap-threshold",
        type=float,
        default=0.5,
        help="Allowed absolute MAE gap between ONNX and TFLite",
    )
    parser.add_argument(
        "--reference-mae",
        type=float,
        default=4.718171434096904,
        help="Reference MAE from training summary",
    )
    parser.add_argument(
        "--training-summary-path",
        default="cluster_results/age-estimation-visualization/tensorboard/utkface/MobileAgeNet/trial_0_20260402-111050/training_summary.json",
        help="Path to training_summary.json to extract best/final val MAE reference",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    summary_best_mae, summary_final_mae = _extract_training_reference_mae(args.training_summary_path)
    if summary_best_mae is not None:
        args.reference_mae = float(summary_best_mae)

    layout, image_h, image_w, input_name = _infer_onnx_layout_and_size(str(onnx_path))
    print(
        f"Loading samples (split={args.eval_split}, n={'all' if args.max_samples == 0 else args.max_samples}, layout={layout}, size={image_h}x{image_w}, input={input_name})..."
    )
    samples = _prepare_samples(args.max_samples, image_h, image_w, args.eval_split, args.split_seed)
    if not samples:
        raise RuntimeError("No samples found for selected split and preprocessing.")

    print("Evaluating ONNX MAE...")
    onnx_mae = _evaluate_onnx_mae(str(onnx_path), samples)

    print("Converting ONNX to TFLite...")
    with tempfile.TemporaryDirectory() as tmpdir:
        _, exported_tflite_path, export_h, export_w, export_layout = _convert_onnx_to_tflite(
            onnx_path=str(onnx_path),
            tflite_path=args.tflite_path,
            workdir=tmpdir,
        )

    print("Verifying exported TFLite input shape...")
    exported_raw_shape, exported_sig_shape, exported_dtype = _verify_exported_tflite_shape(
        exported_tflite_path,
        export_h,
        export_w,
    )

    print("Evaluating TFLite MAE...")
    tflite_mae = _evaluate_tflite_mae(
        exported_tflite_path,
        samples,
        expected_h=export_h,
        expected_w=export_w,
    )

    mae_gap = abs(tflite_mae - onnx_mae)
    ref_gap = abs(tflite_mae - args.reference_mae)
    pass_gap = mae_gap <= args.mae_gap_threshold

    model_name = "MobileAgeNet"
    report = {
        model_name: {
            "status": "success" if pass_gap else "warning_large_gap",
            "onnx_path": str(onnx_path.as_posix()),
            "tflite_path": str(Path(args.tflite_path).as_posix()),
            "tflite_size_mb": round(Path(args.tflite_path).stat().st_size / (1024 * 1024), 4),
            "onnx_mae": round(onnx_mae, 4),
            "tflite_mae": round(tflite_mae, 4),
            "mae_gap_onnx_vs_tflite": round(mae_gap, 4),
            "reference_mae": round(args.reference_mae, 4),
            "mae_gap_vs_reference": round(ref_gap, 4),
            "mae_gap_threshold": round(args.mae_gap_threshold, 4),
            "samples_evaluated": len(samples),
            "eval_split": args.eval_split,
            "split_seed": args.split_seed,
            "training_summary_path": args.training_summary_path,
            "training_summary_best_val_mae": round(float(summary_best_mae), 4) if summary_best_mae is not None else None,
            "training_summary_final_val_mae": round(float(summary_final_mae), 4) if summary_final_mae is not None else None,
            "onnx_layout": export_layout,
            "onnx_input_name": input_name,
            "onnx_spatial_size": [export_h, export_w],
            "tflite_exported_input_shape": exported_raw_shape,
            "tflite_exported_shape_signature": exported_sig_shape,
            "tflite_input_dtype": str(np.dtype(exported_dtype).name),
            "preprocess": {
                "resize": [image_h, image_w],
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            },
        }
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("=== Conversion Summary ===")
    print(f"ONNX MAE   : {onnx_mae:.4f}")
    print(f"TFLite MAE : {tflite_mae:.4f}")
    print(f"MAE Gap    : {mae_gap:.4f} (threshold={args.mae_gap_threshold:.4f})")
    print(f"Reference  : {args.reference_mae:.4f} (gap={ref_gap:.4f})")
    print(f"TFLite exported raw shape: {exported_raw_shape}")
    print(f"TFLite shape signature   : {exported_sig_shape}")
    if summary_best_mae is not None or summary_final_mae is not None:
        print(
            f"Training Summary MAE (best/final val): "
            f"{summary_best_mae if summary_best_mae is not None else 'n/a'} / "
            f"{summary_final_mae if summary_final_mae is not None else 'n/a'}"
        )
    print(f"Report     : {report_path}")

    return 0 if pass_gap else 2


if __name__ == "__main__":
    raise SystemExit(main())
