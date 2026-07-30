#!/bin/bash

cd "$(dirname "$0")/../.."

if [ -d .venv ]; then
    source .venv/bin/activate
fi

set -euo pipefail

# ============================================================
# Usage
# ============================================================
#
# Run every historical record for the selected config:
#
#   ./util/sh/run_layer_analysis.sh
#
# Run one source JSON only:
#
#   SOURCE_JSON=1.json \
#   ./util/sh/run_layer_analysis.sh
#
# Override the default replay length:
#
#   SOURCE_JSON=1.json \
#   BACKFILL_EPOCHS=5 \
#   ./util/sh/run_layer_analysis.sh
#
# Save logs:
#
#   ./util/sh/run_layer_analysis.sh \
#   2>&1 | tee out/training_log.txt
#
# Replay JSON files are written to:
#
#   out/<CONFIG>-layerstats/
#
# Each successfully completed historical record is atomically
# written back immediately after its replay succeeds.
#
# ============================================================

# ============================================================
# IMAGE CLASSIFICATION
# ============================================================

# CONFIG=img-classification_cifar-10_acc_AirNet
# CONFIG=img-classification_cifar-10_acc_AirNext
#CONFIG=img-classification_cifar-10_acc_AlexNet
# CONFIG=img-classification_cifar-10_acc_BagNet
# CONFIG=img-classification_cifar-10_acc_ComplexNet
# CONFIG=img-classification_cifar-10_acc_BayesianNet-1
# CONFIG=img-classification_cifar-10_acc_ConvNeXt
# CONFIG=img-classification_cifar-10_acc_ConvNeXtTransformer
# CONFIG=img-classification_cifar-10_acc_DPN68
# CONFIG=img-classification_cifar-10_acc_DPN107
# CONFIG=img-classification_cifar-10_acc_DPN131
# CONFIG=img-classification_cifar-10_acc_DarkNet
# CONFIG=img-classification_cifar-10_acc_DenseNet
# CONFIG=img-classification_cifar-10_acc_Diffuser
# CONFIG=img-classification_cifar-10_acc_EfficientNet
# CONFIG=img-classification_cifar-10_acc_FractalNet
# CONFIG=img-classification_cifar-10_acc_GoogLeNet
# CONFIG=img-classification_cifar-10_acc_ICNet
# CONFIG=img-classification_cifar-10_acc_InceptionV3-1
# CONFIG=img-classification_cifar-10_acc_MNASNet
# CONFIG=img-classification_cifar-10_acc_MaxVit
# CONFIG=img-classification_cifar-10_acc_MobileNetV2
# CONFIG=img-classification_cifar-10_acc_MobileNetV3
# CONFIG=img-classification_cifar-10_acc_MoE-hetero4-Alex-Dense-Air-Bag
 CONFIG=img-classification_cifar-10_acc_RegNet
# CONFIG=img-classification_cifar-10_acc_ResNet
# CONFIG=img-classification_cifar-10_acc_ShuffleNet
# CONFIG=img-classification_cifar-10_acc_SqueezeNet-1
# CONFIG=img-classification_cifar-10_acc_SwinTransformer
# CONFIG=img-classification_cifar-10_acc_UNet2D
# CONFIG=img-classification_cifar-10_acc_VGG
# CONFIG=img-classification_cifar-10_acc_VisionTransformer

# ============================================================
# IMAGE SEGMENTATION
# ============================================================

# CONFIG=img-segmentation_coco_iou_DeepLabV3-1
# CONFIG=img-segmentation_coco_iou_FCN8s
# CONFIG=img-segmentation_coco_iou_FCN16s
# CONFIG=img-segmentation_coco_iou_FCN32s-1
# CONFIG=img-segmentation_coco_iou_LRASPP
# CONFIG=img-segmentation_coco_iou_UNet-1

# ============================================================
# OBJECT DETECTION
# ============================================================

# CONFIG=obj-detection_coco_map_FasterRCNN
# CONFIG=obj-detection_coco_map_FCOS
# CONFIG=obj-detection_coco_map_RetinaNet
# CONFIG=obj-detection_coco_map_SSDLite

# ============================================================
# TEXT GENERATION
# ============================================================

# CONFIG=txt-generation_wikitext_ppl_RNN
# CONFIG=txt-generation_wikitext_ppl_LSTM

# ============================================================
# IMAGE CAPTIONING
# ============================================================

# CONFIG=img-captioning_coco_bleu4_RESNETLSTM
# CONFIG=img-captioning_coco_bleu4_ResNetTransformer

# ============================================================
# SUPER RESOLUTION
# ============================================================

# CONFIG=img-super-resolution_div2k_psnr_RLFN

SOURCE_JSON="${SOURCE_JSON:-}"
BACKFILL_EPOCHS="${BACKFILL_EPOCHS:-50}"

python - \
    "$CONFIG" \
    "$SOURCE_JSON" \
    "$BACKFILL_EPOCHS" <<'PY'

import copy
import json
import os
import sys
from pathlib import Path
from ab.nn.util.Exception import AccuracyException

from ab.nn.util.Loader import load_dataset
from ab.nn.util.Train import Train
from ab.nn.util.Util import (
    default_epoch_limit_minutes,
    nn_mod,
)
from ab.nn.util.db.Util import get_ab_nn_attr

CONFIG = sys.argv[1]
SOURCE_JSON = sys.argv[2]
BACKFILL_EPOCHS = int(sys.argv[3])

if BACKFILL_EPOCHS < 1:
    raise SystemExit(
        f"BACKFILL_EPOCHS must be positive; got {BACKFILL_EPOCHS}"
    )

def load_records(path):
    """
    Load a JSON file as a list of records.

    Historical files may contain either one dictionary or
    a list of dictionaries.
    """
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        return [data]

    raise SystemExit(
        f"Unsupported JSON shape in {path}: "
        f"{type(data).__name__}"
    )

def split_config(config):
    """
    Example:

        img-classification_cifar-10_acc_AlexNet

    becomes:

        task    = img-classification
        dataset = cifar-10
        metric  = acc
        model   = AlexNet
    """
    parts = config.split("_", 3)

    if len(parts) != 4:
        raise SystemExit(
            "CONFIG must have the form:\n"
            "  task_dataset_metric_model\n\n"
            f"Received:\n  {config}"
        )

    return parts

def get_json_files(config):
    """
    Return all numeric historical JSON files for this config.
    """
    config_dir = (
        Path("ab/nn/stat/train")
        / config
    )

    if not config_dir.is_dir():
        raise SystemExit(
            "Historical configuration directory does not exist:\n"
            f"  {config_dir}"
        )

    if SOURCE_JSON:
        requested = config_dir / SOURCE_JSON

        if not requested.is_file():
            raise SystemExit(
                "Requested source JSON does not exist:\n"
                f"  {requested}"
            )

        return [requested]

    json_files = [
        path
        for path in config_dir.glob("*.json")
        if path.stem.isdigit()
    ]

    if not json_files:
        raise SystemExit(
            "No numeric historical JSON files found in:\n"
            f"  {config_dir}"
        )

    return sorted(
        json_files,
        key=lambda path: int(path.stem),
    )

def get_replay_dir(config):
    """
    Return a readable, stable directory for replay JSON files.

    Example:

        out/img-classification_cifar-10_acc_AlexNet-layerstats/
    """
    replay_dir = (
        Path("out")
        / f"{config}-layerstats"
    )

    replay_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return replay_dir

def clear_replay_json_files(replay_dir):
    """
    Remove only numeric JSON files generated by replay.

    Other files in the output directory are preserved.
    """
    for path in replay_dir.glob("*.json"):
        if path.stem.isdigit():
            path.unlink()

def get_exact_parameters(record, model_name):
    """
    Extract the exact parameters required to replay one record.

    Required parameters are:

        supported_hyperparameters()
        UNION
        {"batch", "transform"}

    No parameters are guessed or sampled.
    """
    supported = set(
        get_ab_nn_attr(
            f"nn.{model_name}",
            "supported_hyperparameters",
        )()
    )

    required = supported | {
        "batch",
        "transform",
    }

    missing = sorted(
        key
        for key in required
        if (
            key not in record
            or record[key] in (None, "")
        )
    )

    if missing:
        raise SystemExit(
            "Refusing to run because this historical record "
            "is missing required exact hyperparameters:\n"
            + "\n".join(
                f"  - {key}"
                for key in missing
            )
        )

    return {
        key: record[key]
        for key in sorted(required)
    }

def select_all_records(config, model_name):
    """
    Select every historical record independently.

    Records are never deduplicated by hyperparameters.
    """
    selected_records = []

    for json_path in get_json_files(config):
        records = load_records(json_path)

        for record_index, record in enumerate(records):
            parameters = get_exact_parameters(
                record,
                model_name,
            )

            selected_records.append(
                {
                    "source_json": json_path,
                    "record_index": record_index,
                    "record": record,
                    "parameters": parameters,
                }
            )

    return sorted(
        selected_records,
        key=lambda item: (
            str(item["source_json"]),
            item["record_index"],
        ),
    )

def run_training(
    task,
    dataset,
    metric,
    model_name,
    parameters,
    epoch_max,
    replay_dir,
):
    """
    Run one replay directly through Train.train_n_eval().

    Replay JSON files are written into replay_dir.
    """
    replay_parameters = copy.deepcopy(
        parameters
    )

    replay_parameters["epoch_max"] = epoch_max

    batch = max(
        1,
        int(replay_parameters["batch"]),
    )

    transform = replay_parameters["transform"]

    (
        out_shape,
        minimum_accuracy,
        train_set,
        test_set,
    ) = load_dataset(
        task,
        dataset,
        transform,
    )

    trainer = Train(
        config=(
            task,
            dataset,
            metric,
            model_name,
        ),
        out_shape=out_shape,
        minimum_accuracy=minimum_accuracy,
        batch=batch,
        nn_module=nn_mod(
            "nn",
            model_name,
        ),
        task=task,
        train_dataset=train_set,
        test_dataset=test_set,
        metric=metric,
        num_workers=replay_parameters.get(
            "num_workers",
            1,
        ),
        prm=replay_parameters,
        save_to_db=True,
        layer_analysis=True,
    )

    return trainer.train_n_eval(
        epoch_max=epoch_max,
        epoch_limit_minutes=default_epoch_limit_minutes,
        save_pth_weights=False,
        save_onnx_weights=False,
        train_set=train_set,
        save_path=replay_dir,
    )

def atomic_write_json(path, records):
    """
    Safely replace one historical JSON file.

    The destination is written to a temporary file first,
    then atomically replaced.
    """
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            indent=4,
            ensure_ascii=False,
        )
        file.write("\n")

    os.replace(
        temporary_path,
        path,
    )

def record_matches_parameters(
    record,
    parameters,
):
    """
    Match a record using all exact hyperparameters.

    UID is deliberately not used as the identity.
    """
    for key, expected in parameters.items():
        if key not in record:
            return False

        if record[key] != expected:
            return False

    return True

def record_fingerprint(record):
    """
    Stable comparison for a complete JSON record.
    """
    return json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

def get_record_epoch_max(record):
    train_stat = record.get(
        "train_stat"
    )

    if isinstance(train_stat, dict):
        epoch_max = train_stat.get(
            "epoch_max"
        )

        if epoch_max is not None:
            return int(epoch_max)

    epoch_max = record.get(
        "epoch_max"
    )

    if epoch_max is not None:
        return int(epoch_max)

    return None

def find_retraining_record(
    generated_records,
    parameters,
    old_fingerprints,
    replay_epoch_max,
):
    """
    Find the generated record for one replay.

    The record must:

      1. have the requested epoch_max,
      2. match all exact hyperparameters,
      3. contain layer_stat,
      4. be new relative to old_fingerprints.
    """
    matches = []

    for record in generated_records:
        record_epoch_max = get_record_epoch_max(
            record
        )

        if record_epoch_max != replay_epoch_max:
            continue

        if not record_matches_parameters(
            record,
            parameters,
        ):
            continue

        layer_stat = record.get(
            "layer_stat"
        )

        if not isinstance(layer_stat, dict):
            continue

        fingerprint = record_fingerprint(
            record
        )

        matches.append(
            {
                "record": record,
                "is_new": (
                    fingerprint
                    not in old_fingerprints
                ),
            }
        )

    if not matches:
        raise RuntimeError(
            "Could not find a generated replay record "
            "with the exact parameters and layer_stat."
        )

    new_matches = [
        item["record"]
        for item in matches
        if item["is_new"]
    ]

    if len(new_matches) == 1:
        return new_matches[0]

    if len(new_matches) > 1:
        raise RuntimeError(
            f"Found {len(new_matches)} generated records "
            "matching the same exact parameters. "
            "Refusing to guess."
        )

    raise RuntimeError(
        "Found matching generated records, but none were "
        "new relative to the replay output."
    )

def analysis_epochs_for(epoch_max):
    """
    Return the layer-analysis checkpoint schedule.

    For 50 epochs:

        1, 2, 3, 4, 5,
        10, 15, 20, 25, 30,
        35, 40, 45, 50
    """
    epochs = list(
        range(
            1,
            min(5, epoch_max) + 1,
        )
    )

    if epoch_max >= 10:
        epochs.extend(
            range(
                10,
                epoch_max + 1,
                5,
            )
        )

    return epochs

task, dataset, metric, model_name = split_config(
    CONFIG
)

selected_records = select_all_records(
    CONFIG,
    model_name,
)

START_RECORD = int(os.environ.get("START_RECORD", "0"))

selected_records = [
    item
    for item in selected_records
    if item["record_index"] >= START_RECORD
]

historical_json_files = get_json_files(
    CONFIG
)

original_records_by_path = {
    path: copy.deepcopy(
        load_records(path)
    )
    for path in historical_json_files
}

pending_records_by_path = copy.deepcopy(
    original_records_by_path
)

changed_paths = set()

print()
print("Selected historical records")
print("===========================")
print(f"CONFIG: {CONFIG}")
print(
    f"Selected record count: "
    f"{len(selected_records)}"
)
print(
    f"Backfill epochs: {BACKFILL_EPOCHS}"
)
print()

for selected in selected_records:
    source_json = selected["source_json"]
    record_index = selected["record_index"]
    parameters = selected["parameters"]

    replay_epoch_max = BACKFILL_EPOCHS
    expected_epochs = analysis_epochs_for(
        replay_epoch_max
    )

    print("Source JSON")
    print("-----------")
    print(f"  {source_json}")

    print("Selected record")
    print("----------------")
    print(
        f"  record index: {record_index}"
    )

    print(
        f"  replay epochs: {replay_epoch_max}"
    )

    print("Exact parameters")
    print("----------------")
    print(
        json.dumps(
            parameters,
            indent=2,
        )
    )
    print()


    replay_dir = get_replay_dir(
        CONFIG
    )

    clear_replay_json_files(
        replay_dir
    )

    print(
        "Running layer analysis for "
        f"{source_json}, record {record_index}..."
    )
    print(
        f"  replay output: {replay_dir}"
    )
    print()

    try:
        run_training(
            task=task,
            dataset=dataset,
            metric=metric,
            model_name=model_name,
            parameters=parameters,
            epoch_max=replay_epoch_max,
            replay_dir=replay_dir,
        )
    except AccuracyException as exc:
        print()
        print(
            "Skipping historical record because "
            "the accuracy/time threshold was not met:"
        )
        print(f"  source JSON: {source_json}")
        print(f"  record index: {record_index}")
        print(f"  reason: {exc}")
        print()
        continue

    final_json = (
        replay_dir
        / f"{replay_epoch_max}.json"
    )


    if not final_json.exists():
        raise RuntimeError(
            "Expected replay JSON file does not exist:\n"
            f"  {final_json}"
        )

    generated_records = load_records(
        final_json
    )

    retrained_record = find_retraining_record(
        generated_records=generated_records,
        parameters=parameters,
        old_fingerprints=set(),
        replay_epoch_max=replay_epoch_max,
    )

    complete_layer_stat = retrained_record.get(
        "layer_stat"
    )

    if not isinstance(
        complete_layer_stat,
        dict,
    ):
        raise RuntimeError(
            "The replay record has no "
            "layer_stat dictionary:\n"
            f"  file: {final_json}\n"
            f"  source JSON: {source_json}\n"
            f"  record index: {record_index}"
        )

    numeric_keys = {
        str(key)
        for key in complete_layer_stat
        if str(key).isdigit()
    }

    expected_keys = {
        str(epoch)
        for epoch in expected_epochs
    }

    if numeric_keys != expected_keys:
        raise RuntimeError(
            "Unexpected cumulative layer_stat keys:\n"
            f"  source JSON: {source_json}\n"
            f"  record index: {record_index}\n"
            f"  replay file: {final_json}\n"
            f"  expected: "
            f"{sorted(expected_keys, key=int)}\n"
            f"  found: "
            f"{sorted(numeric_keys, key=int)}"
        )

    complete_layer_stat = copy.deepcopy(
        complete_layer_stat
    )

    target_records = pending_records_by_path[
        source_json
    ]

    if record_index >= len(target_records):
        raise RuntimeError(
            "Selected record index disappeared from "
            f"{source_json}: {record_index}"
        )

    target_record = target_records[
        record_index
    ]

    if not record_matches_parameters(
        target_record,
        parameters,
    ):
        raise RuntimeError(
            "Historical record changed before writeback:\n"
            f"  source JSON: {source_json}\n"
            f"  record index: {record_index}"
        )

    target_record[
        "layer_stat"
    ] = complete_layer_stat

    changed_paths.add(
        source_json
    )

    atomic_write_json(
        source_json,
        pending_records_by_path[
            source_json
        ],
    )

    print()
    print(
        "Checkpoint writeback completed"
    )
    print(
        "-----------------------------"
    )
    print(
        f"  updated: {source_json}"
    )
    print(
        f"  record index: {record_index}"
    )
    print()

    print()
    print("Complete layer analysis found")
    print("-----------------------------")
    print(
        f"  source JSON: {source_json}"
    )
    print(
        f"  record index: {record_index}"
    )
    print(
        f"  snapshots: "
        f"{sorted(numeric_keys, key=int)}"
    )
    print(
        "  prepared writeback for the original "
        "historical record"
    )
    print()

print()
print(
    "Incremental historical JSON "
    "writeback completed."
)
print(
    "Each completed record was written "
    "atomically."
)
print()
print(
    "All selected layer-analysis replays "
    "completed."
)

PY